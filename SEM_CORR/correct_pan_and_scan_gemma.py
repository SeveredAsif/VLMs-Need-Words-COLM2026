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
import math
import numpy as np
import matplotlib.pyplot as plt
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from data_class import VisionLanguageDataset


DATASET_NAME = args.dataset_name
BASE_MODEL_PATH = f"google/gemma-3-{args.msize}-it"

print(f"Model: {BASE_MODEL_PATH}, Dataset: {DATASET_NAME}")

test_dataset = VisionLanguageDataset(dataset_name=DATASET_NAME)

processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, padding_side='left', do_pan_and_scan=True)
model = Gemma3ForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH, dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True
).cuda().eval()

N_LAYERS = len(model.model.language_model.layers) + 1

# Gemma 3: images resized to 896x896, 16x16 token grid (256 tokens per image)
GRID_H, GRID_W, PROCESSED_H, PROCESSED_W = 16, 16, 896, 896
VISION_START_ID = 255999


def get_pan_and_scan_layout(width, height, processor):
    # Defaults must match Gemma3ProcessorKwargs._defaults in processing_gemma3.py,
    # NOT the image_processor config (which stores None for these fields).
    ip = processor.image_processor
    min_crop_size = getattr(ip, "pan_and_scan_min_crop_size", 256)
    if min_crop_size is None: min_crop_size = 256
    max_num_crops = getattr(ip, "pan_and_scan_max_num_crops", 4)
    if max_num_crops is None: max_num_crops = 4
    min_ratio = getattr(ip, "pan_and_scan_min_ratio_to_activate", 1.2)
    if min_ratio is None: min_ratio = 1.2

    if width >= height:
        if width / height < min_ratio:
            return 1, 1
        num_crops_w = int(math.floor(width / height + 0.5))
        num_crops_w = min(int(math.floor(width / min_crop_size)), num_crops_w)
        num_crops_w = max(2, num_crops_w)
        num_crops_w = min(max_num_crops, num_crops_w)
        num_crops_h = 1
    else:
        if height / width < min_ratio:
            return 1, 1
        num_crops_h = int(math.floor(height / width + 0.5))
        num_crops_h = min(int(math.floor(height / min_crop_size)), num_crops_h)
        num_crops_h = max(2, num_crops_h)
        num_crops_h = min(max_num_crops, num_crops_h)
        num_crops_w = 1

    crop_size_w = int(math.ceil(width / num_crops_w))
    crop_size_h = int(math.ceil(height / num_crops_h))

    if min(crop_size_w, crop_size_h) < min_crop_size:
        return 1, 1

    return num_crops_w, num_crops_h

def bbox_to_token_indices(x_min, y_min, x_max, y_max, crop_width, crop_height,
                          resized_size=896, tokens_per_side=16):
    width_ratio = resized_size / crop_width
    height_ratio = resized_size / crop_height
    patch_size = resized_size / tokens_per_side

    x_min_token = max(0, min(tokens_per_side - 1, int((x_min * width_ratio) // patch_size)))
    x_max_token = max(0, min(tokens_per_side - 1, int(math.ceil((x_max * width_ratio) / patch_size))))
    y_min_token = max(0, min(tokens_per_side - 1, int((y_min * height_ratio) // patch_size)))
    y_max_token = max(0, min(tokens_per_side - 1, int(math.ceil((y_max * height_ratio) / patch_size))))

    return [y * tokens_per_side + x for y in range(y_min_token, max(y_min_token + 1, y_max_token)) 
            for x in range(x_min_token, max(x_min_token + 1, x_max_token))]

def find_gemma3_image_tokens(image_starts, image_resolution, bbox, crop_layout):
    original_width, original_height = image_resolution
    x_min, y_min, x_max, y_max = bbox
    num_crops_w, num_crops_h = crop_layout
    
    token_indices = set()
    
    # 1. Global Crop
    global_indices = bbox_to_token_indices(x_min, y_min, x_max, y_max, original_width, original_height)
    token_indices.update([i + image_starts[0] for i in global_indices])
    
    # 2. Local Crops
    if num_crops_w > 1 or num_crops_h > 1:
        crop_width = math.ceil(original_width / num_crops_w)
        crop_height = math.ceil(original_height / num_crops_h)
        
        crop_idx = 1
        for h in range(num_crops_h):
            for w in range(num_crops_w):
                c_x_min = w * crop_width
                c_y_min = h * crop_height
                c_x_max = c_x_min + crop_width
                c_y_max = c_y_min + crop_height
                
                # Check intersection
                inter_x_min = max(x_min, c_x_min)
                inter_y_min = max(y_min, c_y_min)
                inter_x_max = min(x_max, c_x_max)
                inter_y_max = min(y_max, c_y_max)
                
                if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
                    rel_x_min = inter_x_min - c_x_min
                    rel_y_min = inter_y_min - c_y_min
                    rel_x_max = inter_x_max - c_x_min
                    rel_y_max = inter_y_max - c_y_min
                    
                    local_indices = bbox_to_token_indices(
                        rel_x_min, rel_y_min, rel_x_max, rel_y_max, 
                        crop_width, crop_height
                    )
                    token_indices.update([i + image_starts[crop_idx] for i in local_indices])
                
                crop_idx += 1
                
    return list(token_indices)


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
    inputs = processor([image1, image2], text_input, padding=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs.to(model.device), output_hidden_states=True)
        all_hidden_states = outputs.hidden_states

    answer_key = ord(item["answer"]) - ord("A")

    num_crops_1 = get_pan_and_scan_layout(image1.width, image1.height, processor)
    num_crops_1_total = 1 + (num_crops_1[0] * num_crops_1[1] if num_crops_1 != (1,1) else 0)
    
    num_crops_2 = get_pan_and_scan_layout(image2.width, image2.height, processor)
    num_crops_2_total = 1 + (num_crops_2[0] * num_crops_2[1] if num_crops_2 != (1,1) else 0)
    
    all_image_starts = [i for i, x in enumerate(inputs.input_ids[0].tolist()) if x == VISION_START_ID]
    
    if len(all_image_starts) != num_crops_1_total + num_crops_2_total:
        print(f"WARNING: Expected {num_crops_1_total + num_crops_2_total} crops, found {len(all_image_starts)}")
        print(f"image1 size: {image1.size}, num_crops_1: {num_crops_1}")
        print(f"image2 size: {image2.size}, num_crops_2: {num_crops_2}")
        
        # Fallback to single crop to avoid crashing
        num_crops_1 = (1, 1)
        num_crops_1_total = 1
        num_crops_2 = (1, 1)
        num_crops_2_total = 1
        
    image1_starts = all_image_starts[0:num_crops_1_total]
    image2_starts = all_image_starts[num_crops_1_total : num_crops_1_total + num_crops_2_total]

    # Precompute reference token indices (same across layers)
    ref_box_raw = item["ref_coordinate"]
    ref_original = (
        ref_box_raw[0][0], ref_box_raw[0][1],
        ref_box_raw[1][0], ref_box_raw[1][1]
    )
    ref_indices = find_gemma3_image_tokens(image1_starts, (image1.width, image1.height), ref_original, num_crops_1)

    # Precompute target token indices for each candidate
    tgt_indices_list = []
    for original_box in item["tgt_coordinate"]:
        tgt_original = (
            original_box[0][0], original_box[0][1],
            original_box[1][0], original_box[1][1]
        )
        tgt_indices_list.append(find_gemma3_image_tokens(image2_starts, (image2.width, image2.height), tgt_original, num_crops_2))

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

    os.makedirs(f"rep_faces_results/GEMMA", exist_ok=True)
    with open(f"rep_faces_results/GEMMA/{args.msize}_{DATASET_NAME}_{metric_name}.pkl", "wb") as f:
        pickle.dump(accuracies, f)
    plt.savefig(f"rep_faces_results/GEMMA/{args.msize}_{DATASET_NAME}_{metric_name}.png")
    plt.close()

save_plot(accuracies_maxsim, "maxsim")
save_plot(accuracies_gram, "gram")
