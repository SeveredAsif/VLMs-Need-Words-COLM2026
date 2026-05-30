import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--msize", type=str)
parser.add_argument("--cuda", type=int)
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

import math
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForImageTextToText, AutoProcessor
from tqdm import tqdm
from data_class import VisionLanguageDataset


DATASET_NAME = args.dataset_name
model_path = f"OpenGVLab/InternVL3_5-{args.msize}-HF"

print(f"Model: {model_path}, Dataset: {DATASET_NAME}")

test_dataset = VisionLanguageDataset(dataset_name=DATASET_NAME)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
).cuda().eval()

N_LAYERS = len(model.language_model.layers) + 1

print(f"Number of layers: {N_LAYERS}")

# InternVL3 image processing constants:
#   - 448×448 image tile
#   - 14px ViT patches → 32×32 patches per tile
#   - 2×2 pixel shuffle → 16×16 tokens per tile
#   - Each token covers 28×28 pixels (448/16)
INTERNVL_TILE_SIZE = 448
TOKENS_PER_TILE_SIDE = 16
PX_PER_TOKEN = INTERNVL_TILE_SIZE // TOKENS_PER_TILE_SIDE  # 28

img_context_token_id = processor.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
print(f"Image context token ID: {img_context_token_id}")


def find_consecutive_groups(positions):
    """Group a sorted list of ints into runs of consecutive values."""
    if not positions:
        return []
    groups, current = [], [positions[0]]
    for p in positions[1:]:
        if p == current[-1] + 1:
            current.append(p)
        else:
            groups.append(current)
            current = [p]
    groups.append(current)
    return groups


def infer_spatial_tile_layout(n_spatial_tiles, img_w, img_h):
    """
    Given the number of spatial tiles and the image aspect ratio, pick (tile_h, tile_w)
    whose ratio best matches img_w / img_h (InternVL3 selects tiles this way).
    """
    if n_spatial_tiles == 1:
        return 1, 1
    aspect = img_w / img_h
    best, best_diff = (1, n_spatial_tiles), float('inf')
    for th in range(1, n_spatial_tiles + 1):
        if n_spatial_tiles % th == 0:
            tw = n_spatial_tiles // th
            diff = abs(tw / th - aspect)
            if diff < best_diff:
                best_diff = diff
                best = (th, tw)
    return best  # (tile_h, tile_w)


def find_internvl3_image_tokens(input_ids, bbox, image_index,
                                img_context_token_id,
                                img_w, img_h):
    """
    Map a bounding box (in original image pixel coordinates) to absolute
    token positions in InternVL3-hf's input_ids.
    """
    ids = input_ids.squeeze().tolist()

    img_positions = [i for i, t in enumerate(ids) if t == img_context_token_id]
    groups = find_consecutive_groups(img_positions)
    group = groups[image_index]
    total_tokens = len(group)
    global_start = group[0]

    n_tiles_total = total_tokens // TOKENS_PER_TILE_SIDE ** 2

    if n_tiles_total <= 1:
        canvas_h = canvas_w = INTERNVL_TILE_SIZE
        tile_w = 1
        token_start = global_start
    else:
        n_spatial = n_tiles_total - 1
        tile_h, tile_w = infer_spatial_tile_layout(n_spatial, img_w, img_h)
        canvas_h = tile_h * INTERNVL_TILE_SIZE
        canvas_w = tile_w * INTERNVL_TILE_SIZE
        token_start = global_start

    scale_x = canvas_w / img_w
    scale_y = canvas_h / img_h
    x_min, y_min, x_max, y_max = bbox

    grid_h = (canvas_h // INTERNVL_TILE_SIZE) * TOKENS_PER_TILE_SIDE
    grid_w = (canvas_w // INTERNVL_TILE_SIZE) * TOKENS_PER_TILE_SIDE

    start_gx = max(0, int(x_min * scale_x / PX_PER_TOKEN))
    end_gx   = min(grid_w, math.ceil(x_max * scale_x / PX_PER_TOKEN))
    start_gy = max(0, int(y_min * scale_y / PX_PER_TOKEN))
    end_gy   = min(grid_h, math.ceil(y_max * scale_y / PX_PER_TOKEN))

    tokens = []
    for gy in range(start_gy, end_gy):
        for gx in range(start_gx, end_gx):
            t_row = gy // TOKENS_PER_TILE_SIDE
            t_col = gx // TOKENS_PER_TILE_SIDE
            t_idx = t_row * tile_w + t_col
            l_gy  = gy % TOKENS_PER_TILE_SIDE
            l_gx  = gx % TOKENS_PER_TILE_SIDE
            rel   = t_idx * TOKENS_PER_TILE_SIDE**2 + l_gy * TOKENS_PER_TILE_SIDE + l_gx
            tokens.append(token_start + rel)

    return tokens


def max_sim(ref_features, tgt_features):
    ref_norm = torch.nn.functional.normalize(ref_features, p=2, dim=1)
    tgt_norm = torch.nn.functional.normalize(tgt_features, p=2, dim=1)
    sim_matrix = torch.matmul(ref_norm, tgt_norm.T)
    return torch.mean(torch.max(sim_matrix, dim=1).values).item()


correct_per_layer = [0] * N_LAYERS
n_dataset = len(test_dataset)

for k in tqdm(range(n_dataset)):
    item = test_dataset[k]
    image1 = item["ref_image"]
    image2 = item["tgt_image"]

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image1},
        {"type": "image", "image": image2},
        {"type": "text", "text": item["prompt"]},
    ]}]

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor([[image1, image2]], [text_input], padding=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs.to(model.device), output_hidden_states=True)
        all_hidden_states = outputs.hidden_states

    answer_key = ord(item["answer"]) - ord("A")

    # Precompute reference token indices
    ref_box_raw = item["ref_coordinate"]
    ref_indices = find_internvl3_image_tokens(
        input_ids=inputs.input_ids[0],
        bbox=(ref_box_raw[0][0], ref_box_raw[0][1], ref_box_raw[1][0], ref_box_raw[1][1]),
        image_index=0,
        img_context_token_id=img_context_token_id,
        img_w=image1.width,
        img_h=image1.height,
    )

    # Precompute target token indices for each candidate
    tgt_indices_list = []
    for original_box in item["tgt_coordinate"]:
        tgt_indices = find_internvl3_image_tokens(
            input_ids=inputs.input_ids[0],
            bbox=(original_box[0][0], original_box[0][1], original_box[1][0], original_box[1][1]),
            image_index=1,
            img_context_token_id=img_context_token_id,
            img_w=image2.width,
            img_h=image2.height,
        )
        tgt_indices_list.append(tgt_indices)

    for layer in range(N_LAYERS):
        hidden = all_hidden_states[layer]
        ref_features = hidden[0, ref_indices, :]
        sim_scores = [max_sim(ref_features, hidden[0, tgt_idx, :]) for tgt_idx in tgt_indices_list]
        if np.argmax(sim_scores) == answer_key:
            correct_per_layer[layer] += 1

    torch.cuda.empty_cache()

print("\nResults per layer:")
for layer in range(N_LAYERS):
    accuracy = correct_per_layer[layer] / n_dataset
    print(f"Layer {layer}: {accuracy:.4f} ({correct_per_layer[layer]}/{n_dataset})")

layers = list(range(N_LAYERS))
accuracies = [correct_per_layer[layer] / n_dataset for layer in layers]

plt.plot(layers, accuracies)
plt.xlabel("Layer")
plt.ylabel("Accuracy")
plt.xlim(0, N_LAYERS)
plt.ylim(0, 1)

os.makedirs("rep_faces_results/INTERNVL3_5", exist_ok=True)
with open(f"rep_faces_results/INTERNVL3_5/{args.msize}_{DATASET_NAME}.pkl", "wb") as f:
    pickle.dump(accuracies, f)
plt.savefig(f"rep_faces_results/INTERNVL3_5/{args.msize}_{DATASET_NAME}.png")
