# %%
import argparse
import math
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--ps", type=lambda x: x.lower() in ['true', '1', 'yes'], required=True)
parser.add_argument("--cuda", type=int, required=True)
parser.add_argument("--model_path", type=str, required=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from data_class import VisionLanguageDataset

print(f"Using model: {args.model_path}")

OUTPUT_FILE_NAME = f"rep_probe_results/GEMMA/dataset{args.dataset}_model{args.model_path.replace('/', '_')}"
os.makedirs(os.path.dirname(OUTPUT_FILE_NAME), exist_ok=True)

test_dataset = VisionLanguageDataset(dataset=args.dataset)

DO_PAN_AND_SCAN = args.ps
processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left', do_pan_and_scan=DO_PAN_AND_SCAN)

model = Gemma3ForConditionalGeneration.from_pretrained(
    args.model_path, dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True
).cuda()

model.eval()


N_LAYERS = len(model.model.language_model.layers) + 1


def compute_pan_and_scan_crops(width, height,
                               min_crop_size=256,
                               max_num_crops=4,
                               min_ratio=1.2):
    """
    Replicates Gemma3ImageProcessor.pan_and_scan() to compute crop boundaries.
    Returns a list of (x_min, y_min, x_max, y_max) in original pixel coords.
    Returns empty list if aspect ratio is below threshold (no crops produced).
    """
    if width >= height:
        if width / height < min_ratio:
            return []
        num_crops_w = int(math.floor(width / height + 0.5))
        num_crops_w = min(int(math.floor(width / min_crop_size)), num_crops_w)
        num_crops_w = max(2, min(max_num_crops, num_crops_w))
        num_crops_h = 1 
    else:
        if height / width < min_ratio:
            return []
        num_crops_h = int(math.floor(height / width + 0.5))
        num_crops_h = min(int(math.floor(height / min_crop_size)), num_crops_h)
        num_crops_h = max(2, min(max_num_crops, num_crops_h))
        num_crops_w = 1

    crop_size_w = math.ceil(width / num_crops_w)
    crop_size_h = math.ceil(height / num_crops_h)

    if min(crop_size_w, crop_size_h) < min_crop_size:
        return []

    crops = []
    for h_idx in range(num_crops_h):
        for w_idx in range(num_crops_w):
            x_min = crop_size_w * w_idx
            y_min = crop_size_h * h_idx
            x_max = min(x_min + crop_size_w, width)
            y_max = min(y_min + crop_size_h, height)
            crops.append((x_min, y_min, x_max, y_max))
    return crops


gemma_vision_start_id = 255999
gemma_vision_end_id = 256000


def bbox_to_bbox(original_box):
    """Convert [[x1,y1],[x2,y2]] to (x1, y1, x2, y2)."""
    return (
        original_box[0][0],
        original_box[0][1],
        original_box[1][0],
        original_box[1][1],
    )


def bbox_to_token_indices(x_min, y_min, x_max, y_max, original_width, original_height,
                          resized_size=896, tokens_per_side=16):
    """Convert bounding box to token indices."""
    width_ratio = resized_size / original_width
    height_ratio = resized_size / original_height
    
    x_min_scaled = x_min * width_ratio
    x_max_scaled = x_max * width_ratio
    y_min_scaled = y_min * height_ratio
    y_max_scaled = y_max * height_ratio
    
    patch_size = resized_size / tokens_per_side
    
    x_min_token = int(x_min_scaled // patch_size)
    x_max_token = max(int(x_max_scaled // patch_size), x_min_token + 1)  # at least 1 token wide
    y_min_token = int(y_min_scaled // patch_size)
    y_max_token = max(int(y_max_scaled // patch_size), y_min_token + 1)  # at least 1 token tall

    # Clamp to grid bounds
    x_min_token = max(0, min(x_min_token, tokens_per_side - 1))
    y_min_token = max(0, min(y_min_token, tokens_per_side - 1))
    x_max_token = max(0, min(x_max_token, tokens_per_side))
    y_max_token = max(0, min(y_max_token, tokens_per_side))

    token_indices = []
    for y in range(y_min_token, y_max_token):
        for x in range(x_min_token, x_max_token):
            token_indices.append(y * tokens_per_side + x)

    return token_indices


def get_absolute_token_positions(token_indices, input_ids, vision_start_id=255999, image_index=0):
    """Convert relative token indices to absolute positions in input_ids."""
    image_starts = [i for i, x in enumerate(input_ids[0].tolist()) if x == vision_start_id]
    offset = image_starts[image_index] + 1  # +1 to skip the <start_of_image> marker token
    return [i + offset for i in token_indices]


def find_tokens_in_crops(input_ids, bbox, crops, first_crop_input_ids_idx,
                         vision_start_id=255999, resized_size=896, tokens_per_side=16):
    """
    Map a bounding box to absolute token positions in the pan-and-scan crop tokens.

    Iterates over all crops, finds those that intersect the bbox, transforms the
    intersection into crop-local coordinates, scales to 896x896, and maps to token
    grid indices. Returns absolute positions in input_ids.

    Args:
        input_ids: [1, seq_len] tensor
        bbox: (x_min, y_min, x_max, y_max) in original image pixel coords
        crops: list of (x_min, y_min, x_max, y_max) from compute_pan_and_scan_crops
        first_crop_input_ids_idx: index into image_starts list for crops[0]
                                   (the global view is at first_crop_input_ids_idx - 1)
    Returns:
        list of absolute token positions in input_ids
    """
    image_starts = [i for i, x in enumerate(input_ids[0].tolist()) if x == vision_start_id]
    patch_size = resized_size / tokens_per_side
    ox1, oy1, ox2, oy2 = bbox

    all_positions = []
    for crop_idx, (cx1, cy1, cx2, cy2) in enumerate(crops):
        # Intersection of bbox with this crop
        ix1 = max(ox1, cx1)
        iy1 = max(oy1, cy1)
        ix2 = min(ox2, cx2)
        iy2 = min(oy2, cy2)
        if ix1 >= ix2 or iy1 >= iy2:
            continue

        # Transform intersection to crop-local coords, then scale to resized_size x resized_size
        crop_w = cx2 - cx1
        crop_h = cy2 - cy1
        scale_x = resized_size / crop_w
        scale_y = resized_size / crop_h

        lx1 = (ix1 - cx1) * scale_x
        ly1 = (iy1 - cy1) * scale_y
        lx2 = (ix2 - cx1) * scale_x
        ly2 = (iy2 - cy1) * scale_y

        # Map to token grid
        tx1 = int(lx1 // patch_size)
        ty1 = int(ly1 // patch_size)
        tx2 = max(int(lx2 // patch_size), tx1 + 1)  # at least 1 token wide
        ty2 = max(int(ly2 // patch_size), ty1 + 1)  # at least 1 token tall

        # Clamp to grid bounds
        tx1 = max(0, min(tx1, tokens_per_side - 1))
        ty1 = max(0, min(ty1, tokens_per_side - 1))
        tx2 = max(0, min(tx2, tokens_per_side))
        ty2 = max(0, min(ty2, tokens_per_side))

        token_indices = [y * tokens_per_side + x for y in range(ty1, ty2) for x in range(tx1, tx2)]
        offset = image_starts[first_crop_input_ids_idx + crop_idx] + 1  # +1 to skip the <start_of_image> marker token
        all_positions.extend(t + offset for t in token_indices)

    return all_positions


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
    inputs = processor([image1, image2], text_input, padding=True, return_tensors="pt")

    # Compute pan-and-scan crops for both images (once per sample).
    # input_ids layout when PAS=True:  [ref_global, *ref_crops, tgt_global, *tgt_crops]
    # input_ids layout when PAS=False: [ref_global, tgt_global]
    if DO_PAN_AND_SCAN:
        crops_ref = compute_pan_and_scan_crops(image1.width, image1.height)
        crops_tgt = compute_pan_and_scan_crops(image2.width, image2.height)
        first_crop_ref_idx = 1                        # ref global at slot 0, crops start at slot 1
        first_crop_tgt_idx = 1 + len(crops_ref) + 1  # tgt global at slot (1+N_ref), crops after
    else:
        crops_ref = []
        crops_tgt = []

    # Get all hidden states ONCE per sample
    with torch.no_grad():
        outputs = model(**inputs.to(model.device), output_hidden_states=True)
        all_hidden_states = outputs.hidden_states

    answer_key = ord(test_dataset[k]["answer"]) - ord("A")

    for layer in range(N_LAYERS):
        layer_hidden_state = all_hidden_states[layer]

        target_representations = []
        for original_box in test_dataset[k]["tgt_positions"]:
            bbox = bbox_to_bbox(original_box)
            if DO_PAN_AND_SCAN and crops_tgt:
                target_indices = find_tokens_in_crops(
                    input_ids=inputs.input_ids,
                    bbox=bbox,
                    crops=crops_tgt,
                    first_crop_input_ids_idx=first_crop_tgt_idx,
                )
            else:
                rel = bbox_to_token_indices(bbox[0], bbox[1], bbox[2], bbox[3], image2.width, image2.height)
                target_indices = get_absolute_token_positions(rel, inputs.input_ids, image_index=1)
            region_features = layer_hidden_state[0, target_indices, :]
            target_representations.append(region_features)

        reference_representations = []
        for original_box in test_dataset[k]["ref_positions"]:
            bbox = bbox_to_bbox(original_box)
            if DO_PAN_AND_SCAN and crops_ref:
                ref_indices = find_tokens_in_crops(
                    input_ids=inputs.input_ids,
                    bbox=bbox,
                    crops=crops_ref,
                    first_crop_input_ids_idx=first_crop_ref_idx,
                )
            else:
                rel = bbox_to_token_indices(bbox[0], bbox[1], bbox[2], bbox[3], image1.width, image1.height)
                ref_indices = get_absolute_token_positions(rel, inputs.input_ids, image_index=0)
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

