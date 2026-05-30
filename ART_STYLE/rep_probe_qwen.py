import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--msize", type=str)
parser.add_argument("--cuda", type=int)
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from data_class import VisionLanguageDataset


DATASET_NAME = args.dataset_name
BASE_MODEL_PATH = f"Qwen/Qwen3-VL-{args.msize}-Instruct"

print(f"Model: {BASE_MODEL_PATH}, Dataset: {DATASET_NAME}")

test_dataset = VisionLanguageDataset(dataset_name=DATASET_NAME)

processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, padding_side='left')
model = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH, dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True
).cuda().eval()

N_LAYERS = len(model.model.language_model.layers) + 1


def find_qwen3vl_image_tokens(input_ids, image_resolution, bbox, image_index=0,
                               vision_start_id=151652, vision_end_id=151653):
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()

    start_indices = [i for i, t in enumerate(input_ids) if t == vision_start_id]
    if image_index >= len(start_indices):
        raise ValueError(f"Image index {image_index} out of range. Found {len(start_indices)} images.")

    global_start_idx = start_indices[image_index] + 1
    global_end_idx = input_ids.index(vision_end_id, global_start_idx)
    num_actual_tokens = global_end_idx - global_start_idx

    EFFECTIVE_BLOCK_SIZE = 32  # 16px patch * 2 spatial merge

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


def gram_sim(ref_features, tgt_features):
    ref_norm = torch.nn.functional.normalize(ref_features, p=2, dim=1)
    tgt_norm = torch.nn.functional.normalize(tgt_features, p=2, dim=1)
    
    ref_gram = torch.matmul(ref_norm.T, ref_norm) / ref_features.shape[0]
    tgt_gram = torch.matmul(tgt_norm.T, tgt_norm) / tgt_features.shape[0]
    
    ref_gram_flat = ref_gram.flatten().unsqueeze(0)
    tgt_gram_flat = tgt_gram.flatten().unsqueeze(0)
    
    return -torch.nn.functional.mse_loss(ref_gram_flat, tgt_gram_flat).item()


correct_per_layer_maxsim = [0] * (N_LAYERS)
correct_per_layer_gram = [0] * (N_LAYERS)
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

    # Precompute reference token indices
    ref_grid_thw = inputs.image_grid_thw[0]
    ref_processed_h, ref_processed_w = int(ref_grid_thw[1]) * 16, int(ref_grid_thw[2]) * 16
    ref_box_raw = item["ref_coordinate"]
    ref_scaled = (
        int(ref_box_raw[0][0] * ref_processed_w / image1.width),
        int(ref_box_raw[0][1] * ref_processed_h / image1.height),
        int(ref_box_raw[1][0] * ref_processed_w / image1.width),
        int(ref_box_raw[1][1] * ref_processed_h / image1.height),
    )
    ref_indices = find_qwen3vl_image_tokens(inputs.input_ids[0], (ref_processed_h, ref_processed_w), ref_scaled, image_index=0)

    # Precompute target token indices for each candidate
    tgt_grid_thw = inputs.image_grid_thw[1]
    tgt_processed_h, tgt_processed_w = int(tgt_grid_thw[1]) * 16, int(tgt_grid_thw[2]) * 16
    tgt_indices_list = []
    for original_box in item["tgt_coordinate"]:
        tgt_scaled = (
            int(original_box[0][0] * tgt_processed_w / image2.width),
            int(original_box[0][1] * tgt_processed_h / image2.height),
            int(original_box[1][0] * tgt_processed_w / image2.width),
            int(original_box[1][1] * tgt_processed_h / image2.height),
        )
        tgt_indices_list.append(find_qwen3vl_image_tokens(inputs.input_ids[0], (tgt_processed_h, tgt_processed_w), tgt_scaled, image_index=1))

    for layer in range(N_LAYERS):
        hidden = all_hidden_states[layer]
        ref_features = hidden[0, ref_indices, :]
        
        sim_scores_maxsim = [max_sim(ref_features, hidden[0, tgt_idx, :]) for tgt_idx in tgt_indices_list]
        if np.argmax(sim_scores_maxsim) == answer_key:
            correct_per_layer_maxsim[layer] += 1
            
        sim_scores_gram = [gram_sim(ref_features, hidden[0, tgt_idx, :]) for tgt_idx in tgt_indices_list]
        if np.argmax(sim_scores_gram) == answer_key:
            correct_per_layer_gram[layer] += 1

    torch.cuda.empty_cache()

print("\nResults per layer (MaxSim):")
for layer in range(N_LAYERS):
    accuracy = correct_per_layer_maxsim[layer] / n_dataset
    print(f"Layer {layer}: {accuracy:.4f} ({correct_per_layer_maxsim[layer]}/{n_dataset})")

print("\nResults per layer (Gram):")
for layer in range(N_LAYERS):
    accuracy = correct_per_layer_gram[layer] / n_dataset
    print(f"Layer {layer}: {accuracy:.4f} ({correct_per_layer_gram[layer]}/{n_dataset})")

layers = list(range(N_LAYERS))
accuracies_maxsim = [correct_per_layer_maxsim[layer] / n_dataset for layer in layers]
accuracies_gram = [correct_per_layer_gram[layer] / n_dataset for layer in layers]

def save_plot(accuracies, metric_name):
    plt.figure()
    plt.plot(layers, accuracies)
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.xlim(0, N_LAYERS)
    plt.ylim(0, 1)

    best_layer = int(np.argmax(accuracies))
    best_accuracy = accuracies[best_layer]
    plt.plot(best_layer, best_accuracy, 'ro')
    y_offset = 10 if best_accuracy < 0.9 else -15
    plt.annotate(f'Best: L{best_layer} ({best_accuracy:.4f})', 
                 xy=(best_layer, best_accuracy), 
                 xytext=(0, y_offset),
                 textcoords='offset points',
                 ha='center')

    os.makedirs("rep_faces_results/QWEN", exist_ok=True)
    with open(f"rep_faces_results/QWEN/{args.msize}_{DATASET_NAME}_{metric_name}.pkl", "wb") as f:
        pickle.dump(accuracies, f)
    plt.savefig(f"rep_faces_results/QWEN/{args.msize}_{DATASET_NAME}_{metric_name}.png")
    plt.close()

save_plot(accuracies_maxsim, "maxsim")
save_plot(accuracies_gram, "gram")
