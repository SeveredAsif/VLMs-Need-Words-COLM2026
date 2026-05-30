import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--msize", type=str)
parser.add_argument("--cuda", type=int)
args = parser.parse_args()

OUTPUT_DIR = "rep_faces_results/QWEN"
DATASET_NAME = args.dataset_name.replace("/", "")
BASE_MODEL_PATH = f"Qwen/Qwen3-VL-{args.msize}-Instruct"
CHECKPOINT_PATH = BASE_MODEL_PATH

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from data_class import VisionLanguageDataset

print(f"Model: {CHECKPOINT_PATH}, Dataset: {DATASET_NAME}")
test_dataset = VisionLanguageDataset(dataset_name=DATASET_NAME)
processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, padding_side='left')

model = Qwen3VLForConditionalGeneration.from_pretrained(
    CHECKPOINT_PATH, dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True
).cuda().eval()

N_LAYERS = len(model.model.language_model.layers) + 1

# Qwen3-VL: 16px patch with 2x spatial merge -> 32px effective block
PATCH_SIZE = 16
EFFECTIVE_BLOCK_SIZE = 32
VISION_START_ID = 151652
VISION_END_ID = 151653


def processed_dims(grid_thw):
    return int(grid_thw[1]) * PATCH_SIZE, int(grid_thw[2]) * PATCH_SIZE


def scale_bbox(box_raw, processed_w, processed_h, image_w, image_h):
    scale_x = processed_w / image_w
    scale_y = processed_h / image_h
    return (
        int(box_raw[0][0] * scale_x), int(box_raw[0][1] * scale_y),
        int(box_raw[1][0] * scale_x), int(box_raw[1][1] * scale_y),
    )


def find_qwen3vl_image_tokens(input_ids, image_resolution, bbox, image_index=0):
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()

    start_indices = [i for i, t in enumerate(input_ids) if t == VISION_START_ID]
    if image_index >= len(start_indices):
        raise ValueError(f"Image index {image_index} out of range. Found {len(start_indices)} images.")

    global_start_idx = start_indices[image_index] + 1
    global_end_idx = input_ids.index(VISION_END_ID, global_start_idx)
    num_actual_tokens = global_end_idx - global_start_idx

    height, width = image_resolution
    grid_h = height // EFFECTIVE_BLOCK_SIZE
    grid_w = width // EFFECTIVE_BLOCK_SIZE

    if grid_h * grid_w != num_actual_tokens:
        raise ValueError(
            f"Resolution mismatch. {height}x{width} implies {grid_h}x{grid_w}={grid_h*grid_w} tokens, "
            f"but found {num_actual_tokens} tokens in input_ids."
        )

    x1, y1, x2, y2 = bbox
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    start_gx = max(0, int(x_min / EFFECTIVE_BLOCK_SIZE))
    end_gx = min(grid_w, int((x_max + EFFECTIVE_BLOCK_SIZE - 1) / EFFECTIVE_BLOCK_SIZE))
    start_gy = max(0, int(y_min / EFFECTIVE_BLOCK_SIZE))
    end_gy = min(grid_h, int((y_max + EFFECTIVE_BLOCK_SIZE - 1) / EFFECTIVE_BLOCK_SIZE))

    token_indices = []
    for gy in range(start_gy, end_gy):
        for gx in range(start_gx, end_gx):
            px_start, px_end = gx * EFFECTIVE_BLOCK_SIZE, (gx + 1) * EFFECTIVE_BLOCK_SIZE
            py_start, py_end = gy * EFFECTIVE_BLOCK_SIZE, (gy + 1) * EFFECTIVE_BLOCK_SIZE
            if px_start < x_max and px_end > x_min and py_start < y_max and py_end > y_min:
                token_indices.append(global_start_idx + gy * grid_w + gx)

    return sorted(token_indices)


def max_sim(ref_features, tgt_features):
    ref_norm = torch.nn.functional.normalize(ref_features, p=2, dim=1)
    tgt_norm = torch.nn.functional.normalize(tgt_features, p=2, dim=1)
    sim_matrix = torch.matmul(ref_norm, tgt_norm.T)
    return torch.mean(torch.max(sim_matrix, dim=1).values).item()


correct_per_layer = [0] * (N_LAYERS)
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
    inputs = processor(text=[text_input], images=[image1, image2], padding=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs.to(model.device), output_hidden_states=True)
        all_hidden_states = outputs.hidden_states

    answer_key = ord(item["answer"]) - ord("A")

    # Precompute reference token indices (same across layers)
    ref_processed_h, ref_processed_w = processed_dims(inputs.image_grid_thw[0])
    ref_scaled = scale_bbox(item["ref_coordinate"], ref_processed_w, ref_processed_h, image1.width, image1.height)
    ref_indices = find_qwen3vl_image_tokens(
        inputs.input_ids[0], (ref_processed_h, ref_processed_w), ref_scaled, image_index=0
    )

    # Precompute target token indices for each candidate
    tgt_processed_h, tgt_processed_w = processed_dims(inputs.image_grid_thw[1])
    tgt_indices_list = []
    for original_box in item["tgt_coordinate"]:
        tgt_scaled = scale_bbox(original_box, tgt_processed_w, tgt_processed_h, image2.width, image2.height)
        tgt_indices_list.append(find_qwen3vl_image_tokens(
            inputs.input_ids[0], (tgt_processed_h, tgt_processed_w), tgt_scaled, image_index=1
        ))

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

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(f"{OUTPUT_DIR}/{args.msize}_{DATASET_NAME}.pkl", "wb") as f:
    pickle.dump(accuracies, f)
plt.savefig(f"{OUTPUT_DIR}/{args.msize}_{DATASET_NAME}.png")
