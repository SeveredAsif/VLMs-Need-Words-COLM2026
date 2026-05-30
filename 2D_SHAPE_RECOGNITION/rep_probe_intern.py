# %%
import argparse
import math
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--cuda", type=int, required=True)
parser.add_argument("--model_path", type=str, required=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForImageTextToText, AutoProcessor
from tqdm import tqdm
from data_class import VisionLanguageDataset

print(f"Using model: {args.model_path}")

OUTPUT_FILE_NAME = f"rep_probe_results/INTERNVL3_5/dataset{args.dataset}_model{args.model_path.replace('/', '_')}"
os.makedirs(os.path.dirname(OUTPUT_FILE_NAME), exist_ok=True)

test_dataset = VisionLanguageDataset(dataset=args.dataset)

processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left')

model = AutoModelForImageTextToText.from_pretrained(
    args.model_path,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
).cuda()

model.eval()

N_LAYERS = len(model.language_model.layers) + 1

# InternVL3 image processing constants:
#   - 448×448 image tile
#   - 14px ViT patches → 32×32 patches per tile
#   - 2×2 pixel shuffle → 16×16 tokens per tile
#   - Each token covers 28×28 pixels (448/16)
INTERNVL_TILE_SIZE = 448
TOKENS_PER_TILE_SIDE = 16
PX_PER_TOKEN = INTERNVL_TILE_SIZE // TOKENS_PER_TILE_SIDE  # 28

img_context_token_id = processor.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')


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


def bbox_to_bbox(original_box):
    """Convert [[x1,y1],[x2,y2]] to (x1, y1, x2, y2)."""
    return (
        original_box[0][0],
        original_box[0][1],
        original_box[1][0],
        original_box[1][1],
    )


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


def max_pool_cos_sim(ref_features, tgt_features):
    """Max pool both feature sets, then compute cosine similarity."""
    ref_pooled = torch.max(ref_features, dim=0).values
    tgt_pooled = torch.max(tgt_features, dim=0).values
    cos_sim = torch.nn.functional.cosine_similarity(ref_pooled, tgt_pooled, dim=0)
    return cos_sim.item()


def max_sim(ref_features, tgt_features):
    """ColBERT-style MaxSim: for each reference token, find max sim with any target token, then average."""
    ref_norm = torch.nn.functional.normalize(ref_features, p=2, dim=1)
    tgt_norm = torch.nn.functional.normalize(tgt_features, p=2, dim=1)
    sim_matrix = torch.matmul(ref_norm, tgt_norm.T)
    max_sims = torch.max(sim_matrix, dim=1).values
    return torch.mean(max_sims).item()


# Initialize counters for all layers
correct_per_layer = [0 for _ in range(N_LAYERS)]

n_dataset = len(test_dataset)

for k in tqdm(range(n_dataset)):

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_dataset[k]["ref_image"]},
                {"type": "image", "image": test_dataset[k]["tgt_image"]},
                {"type": "text", "text": test_dataset[k]["prompt"]},
            ],
        }
    ]

    image1 = test_dataset[k]["ref_image"]
    image2 = test_dataset[k]["tgt_image"]

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor([[image1, image2]], [text_input], padding=True, return_tensors="pt")

    # Get all hidden states ONCE per sample
    with torch.no_grad():
        outputs = model(**inputs.to(model.device), output_hidden_states=True)
        all_hidden_states = outputs.hidden_states

    answer_key = ord(test_dataset[k]["answer"]) - ord("A")

    for layer in range(N_LAYERS):
        layer_hidden_state = all_hidden_states[layer]

        target_representations = []
        for original_box in test_dataset[k]["tgt_positions"]:
            target_indices = find_internvl3_image_tokens(
                input_ids=inputs.input_ids[0],
                bbox=bbox_to_bbox(original_box),
                image_index=1,
                img_context_token_id=img_context_token_id,
                img_w=image2.width,
                img_h=image2.height,
            )
            region_features = layer_hidden_state[0, target_indices, :]
            target_representations.append(region_features)

        reference_representations = []
        for original_box in test_dataset[k]["ref_positions"]:
            ref_indices = find_internvl3_image_tokens(
                input_ids=inputs.input_ids[0],
                bbox=bbox_to_bbox(original_box),
                image_index=0,
                img_context_token_id=img_context_token_id,
                img_w=image1.width,
                img_h=image1.height,
            )
            region_features = layer_hidden_state[0, ref_indices, :]
            reference_representations.append(region_features)

        similarity_fn = max_sim

        ref_rep = reference_representations[answer_key]
        sim_scores = [similarity_fn(ref_rep, tgt_rep) for tgt_rep in target_representations]

        if np.argmax(sim_scores) == answer_key:
            correct_per_layer[layer] += 1

    torch.cuda.empty_cache()

# Print results for all layers
print("\nResults per layer:")
for layer in range(N_LAYERS):
    accuracy = correct_per_layer[layer] / n_dataset
    print(f"Layer {layer}: {accuracy:.4f} ({correct_per_layer[layer]}/{n_dataset})")


# Plot accuracy vs layer
layers = list(range(N_LAYERS))
accuracies = [correct_per_layer[layer] / n_dataset for layer in layers]

print(accuracies)
plt.plot(layers, accuracies)
plt.xlabel("Layer")
plt.ylabel("Accuracy")
plt.xlim(0, N_LAYERS)
plt.ylim(0, 1)
max_acc = max(accuracies)
max_acc_layer = accuracies.index(max_acc)
plt.axhline(y=max_acc, alpha=0.5, color='red', linestyle='--', label=f'Max Accuracy: {max_acc:.4f} at Layer {max_acc_layer}')
plt.legend()
# plt.show()

with open(f"{OUTPUT_FILE_NAME}.pkl", "wb") as f:
    pickle.dump(accuracies, f)

plt.savefig(f"{OUTPUT_FILE_NAME}.png")
