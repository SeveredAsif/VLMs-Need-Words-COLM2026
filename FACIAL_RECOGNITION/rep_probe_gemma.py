import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--msize", type=str)
parser.add_argument("--cuda", type=int)
args = parser.parse_args()

OUTPUT_DIR = f"rep_probe_results/GEMMA"
DATASET_NAME = args.dataset_name.replace("/", "")
BASE_MODEL_PATH = f"google/gemma-3-{args.msize}-it"
CHECKPOINT_PATH = BASE_MODEL_PATH


import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from data_class import VisionLanguageDataset

print(f"Model: {CHECKPOINT_PATH}, Dataset: {DATASET_NAME}")
test_dataset = VisionLanguageDataset(dataset_name=DATASET_NAME)
processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, padding_side='left')

model = Gemma3ForConditionalGeneration.from_pretrained(
    CHECKPOINT_PATH, dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True
).cuda().eval()

N_LAYERS = len(model.model.language_model.layers) + 1

# Gemma 3: images resized to 896x896, 16x16 token grid (256 tokens per image)
GRID_H, GRID_W, PROCESSED_H, PROCESSED_W = 16, 16, 896, 896
VISION_START_ID = 255999


def bbox_to_token_indices(x_min, y_min, x_max, y_max, original_width, original_height,
                          resized_size=896, tokens_per_side=16):
    width_ratio = resized_size / original_width
    height_ratio = resized_size / original_height
    patch_size = resized_size / tokens_per_side

    x_min_token = int((x_min * width_ratio) // patch_size)
    x_max_token = int((x_max * width_ratio) // patch_size)
    y_min_token = int((y_min * height_ratio) // patch_size)
    y_max_token = int((y_max * height_ratio) // patch_size)

    return [y * tokens_per_side + x for y in range(y_min_token, y_max_token) for x in range(x_min_token, x_max_token)]


def get_absolute_token_positions(token_indices, input_ids, image_index=0):
    image_starts = [i for i, x in enumerate(input_ids[0].tolist()) if x == VISION_START_ID]
    offset = image_starts[image_index]
    return [i + offset for i in token_indices]


def find_gemma3_image_tokens(input_ids, image_resolution, bbox, image_index=0):
    x_min, y_min, x_max, y_max = bbox
    original_width, original_height = image_resolution
    token_indices = bbox_to_token_indices(x_min, y_min, x_max, y_max, original_width, original_height)
    return get_absolute_token_positions(token_indices, input_ids, image_index)


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
    inputs = processor([image1, image2], text_input, padding=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs.to(model.device), output_hidden_states=True)
        all_hidden_states = outputs.hidden_states

    answer_key = ord(item["answer"]) - ord("A")

    # Precompute reference token indices (same across layers)
    ref_box_raw = item["ref_coordinate"]
    scale_x = PROCESSED_W / image1.width
    scale_y = PROCESSED_H / image1.height
    ref_scaled = (
        int(ref_box_raw[0][0] * scale_x), int(ref_box_raw[0][1] * scale_y),
        int(ref_box_raw[1][0] * scale_x), int(ref_box_raw[1][1] * scale_y),
    )
    ref_indices = find_gemma3_image_tokens(inputs.input_ids, (PROCESSED_H, PROCESSED_W), ref_scaled, image_index=0)

    # Precompute target token indices for each candidate
    tgt_indices_list = []
    scale_x = PROCESSED_W / image2.width
    scale_y = PROCESSED_H / image2.height
    for original_box in item["tgt_coordinate"]:
        tgt_scaled = (
            int(original_box[0][0] * scale_x), int(original_box[0][1] * scale_y),
            int(original_box[1][0] * scale_x), int(original_box[1][1] * scale_y),
        )
        tgt_indices_list.append(find_gemma3_image_tokens(inputs.input_ids, (PROCESSED_H, PROCESSED_W), tgt_scaled, image_index=1))

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
