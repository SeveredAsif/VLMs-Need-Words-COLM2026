# %%
import argparse
import math
import os
import pickle
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--ps", type=lambda x: x.lower() in ['true', '1', 'yes'], required=True)
parser.add_argument("--cuda", type=int, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--sample_size", type=int, default=-1)
parser.add_argument("--ignore_colors", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True)
parser.add_argument("--ignore_options", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

import torch
import numpy as np
import matplotlib.pyplot as plt
import webcolors
from torch.utils.data import DataLoader, Subset
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from data_class import VisionLanguageDataset

IGNORE_COLORS = args.ignore_colors
IGNORE_OPTIONS = args.ignore_options
DO_PAN_AND_SCAN = args.ps

print(f"Using model: {args.model_path}")

OUTPUT_FILE_NAME = f"logit_lens_results/GEMMA/dataset{args.dataset}_model{args.model_path.replace('/', '_')}"
os.makedirs(os.path.dirname(OUTPUT_FILE_NAME), exist_ok=True)

test_dataset = VisionLanguageDataset(dataset=args.dataset)

processor = AutoProcessor.from_pretrained(
    args.model_path, trust_remote_code=True, padding_side='left', do_pan_and_scan=DO_PAN_AND_SCAN
)
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
    Returns empty list if aspect ratio is below threshold.
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

    crop_size_w = math.ceil(width  / num_crops_w)
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
    """Convert bounding box to token indices in the 16x16 image token grid."""
    width_ratio  = resized_size / original_width
    height_ratio = resized_size / original_height

    patch_size = resized_size / tokens_per_side

    x_min_token = int(x_min * width_ratio  // patch_size)
    x_max_token = max(int(x_max * width_ratio  // patch_size), x_min_token + 1)  # at least 1 wide
    y_min_token = int(y_min * height_ratio // patch_size)
    y_max_token = max(int(y_max * height_ratio // patch_size), y_min_token + 1)  # at least 1 tall

    # Clamp to grid
    x_min_token = max(0, min(x_min_token, tokens_per_side - 1))
    y_min_token = max(0, min(y_min_token, tokens_per_side - 1))
    x_max_token = max(0, min(x_max_token, tokens_per_side))
    y_max_token = max(0, min(y_max_token, tokens_per_side))

    return [y * tokens_per_side + x
            for y in range(y_min_token, y_max_token)
            for x in range(x_min_token, x_max_token)]


def get_absolute_token_positions(token_indices, input_ids, vision_start_id=255999, image_index=0):
    """Convert relative token indices to absolute positions in input_ids."""
    image_starts = [i for i, x in enumerate(input_ids[0].tolist()) if x == vision_start_id]
    offset = image_starts[image_index] + 1  # +1 to skip past the <image_start> token itself
    return [i + offset for i in token_indices]


def find_tokens_in_crops(input_ids, bbox, crops, first_crop_input_ids_idx,
                         vision_start_id=255999, resized_size=896, tokens_per_side=16):
    """
    Map a bounding box to absolute token positions across pan-and-scan crop tokens.

    For each crop that intersects the bbox, transforms the intersection into
    crop-local coordinates, scales to 896x896, and maps to the token grid.

    Args:
        input_ids: [1, seq_len] for this sample
        bbox: (x_min, y_min, x_max, y_max) in original image pixel coords
        crops: list of (x_min, y_min, x_max, y_max) from compute_pan_and_scan_crops
        first_crop_input_ids_idx: index into image_starts for crops[0]
    """
    image_starts = [i for i, x in enumerate(input_ids[0].tolist()) if x == vision_start_id]
    patch_size = resized_size / tokens_per_side
    ox1, oy1, ox2, oy2 = bbox

    all_positions = []
    for crop_idx, (cx1, cy1, cx2, cy2) in enumerate(crops):
        ix1 = max(ox1, cx1);  iy1 = max(oy1, cy1)
        ix2 = min(ox2, cx2);  iy2 = min(oy2, cy2)
        if ix1 >= ix2 or iy1 >= iy2:
            continue

        crop_w = cx2 - cx1;  crop_h = cy2 - cy1
        scale_x = resized_size / crop_w
        scale_y = resized_size / crop_h

        lx1 = (ix1 - cx1) * scale_x;  ly1 = (iy1 - cy1) * scale_y
        lx2 = (ix2 - cx1) * scale_x;  ly2 = (iy2 - cy1) * scale_y

        tx1 = int(lx1 // patch_size)
        ty1 = int(ly1 // patch_size)
        tx2 = max(int(lx2 // patch_size), tx1 + 1)
        ty2 = max(int(ly2 // patch_size), ty1 + 1)

        tx1 = max(0, min(tx1, tokens_per_side - 1))
        ty1 = max(0, min(ty1, tokens_per_side - 1))
        tx2 = max(0, min(tx2, tokens_per_side))
        ty2 = max(0, min(ty2, tokens_per_side))

        token_indices = [y * tokens_per_side + x for y in range(ty1, ty2) for x in range(tx1, tx2)]
        offset = image_starts[first_crop_input_ids_idx + crop_idx] + 1  # +1 to skip <image_start>
        all_positions.extend(t + offset for t in token_indices)

    return all_positions


def get_decoded_tokens(item, k, sample_input_ids, sample_hidden_state,
                       crops_tgt, first_crop_tgt_idx, tgt_global_img_idx):
    """
    Project hidden states for bbox k of item's target image through the lm_head.
    Uses pan-and-scan crop tokens when available, falls back to global view otherwise.

    Args:
        item: dataset item dict
        k: which target bbox (0-3)
        sample_input_ids: [1, seq_len] for this sample (may be padded)
        sample_hidden_state: [1, seq_len, D] hidden states at one layer
        crops_tgt: list of crop bboxes for the target image (may be empty)
        first_crop_tgt_idx: index into image_starts for tgt crops[0]
        tgt_global_img_idx: index into image_starts for tgt global view
    """
    original_box = item["tgt_positions"][k]
    bbox = bbox_to_bbox(original_box)
    tgt_image = item["tgt_image"]

    if DO_PAN_AND_SCAN and crops_tgt:
        target_indices = find_tokens_in_crops(
            input_ids=sample_input_ids,
            bbox=bbox,
            crops=crops_tgt,
            first_crop_input_ids_idx=first_crop_tgt_idx,
        )
    else:
        # Near-square image: PAS produced no crops, fall back to global view
        rel = bbox_to_token_indices(bbox[0], bbox[1], bbox[2], bbox[3],
                                    tgt_image.width, tgt_image.height)
        target_indices = get_absolute_token_positions(rel, sample_input_ids,
                                                       image_index=tgt_global_img_idx)

    region_features = sample_hidden_state[0, target_indices, :]

    with torch.no_grad():
        normalized = model.model.language_model.norm(region_features)
        logits     = model.lm_head(normalized)
        probs      = torch.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=5, dim=-1)

    token_values = []
    for t in range(region_features.shape[0]):
        tok = processor.tokenizer.decode([top_ids[t, 0].item()]).strip()
        token_values.append((tok, top_probs[t, 0].item()))
    return token_values


def collate_fn(batch):
    images_list   = [[item["ref_image"], item["tgt_image"]] for item in batch]
    messages_list = [[
        {"role": "user", "content": [
            {"type": "image", "image": item["ref_image"]},
            {"type": "image", "image": item["tgt_image"]},
            {"type": "text",  "text":  item["prompt"]},
        ]}
    ] for item in batch]
    texts  = processor.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True)
    inputs = processor(images_list, texts, padding=True, return_tensors="pt")
    return inputs, batch


def is_color(name):
    try:
        webcolors.name_to_hex(name)
        return True
    except ValueError:
        return False


def filter_tokens(tokens):
    """Drop color names and/or option letters based on IGNORE_COLORS / IGNORE_OPTIONS flags."""
    result = []
    for t in tokens:
        t_stripped = t.strip()
        if IGNORE_COLORS  and is_color(t_stripped.lower()): continue
        if IGNORE_OPTIONS and t_stripped in {'A', 'B', 'C', 'D'}:  continue
        result.append(t)
    return result


def jaccard_similarity(set1, set2):
    s1, s2 = set(set1), set(set2)
    inter  = len(s1 & s2)
    union  = len(s1 | s2)
    return inter / union if union > 0 else 0.0


def get_logit_metrics(model, target_dataset, num_samples=-1, batch_size=16):
    if num_samples == -1:
        num_samples = len(target_dataset)

    subset     = Subset(target_dataset, range(num_samples))
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    per_layer_jac_scores = defaultdict(list)
    all_data = []

    for inputs, meta_batch in tqdm(dataloader):
        inputs = inputs.to(model.device)
        B = len(meta_batch)

        # Compute PAS crops per sample.
        # input_ids layout: [ref_global, *ref_crops, tgt_global, *tgt_crops]
        crops_info = []
        for item in meta_batch:
            if DO_PAN_AND_SCAN:
                crops_ref = compute_pan_and_scan_crops(item["ref_image"].width, item["ref_image"].height)
                crops_tgt = compute_pan_and_scan_crops(item["tgt_image"].width, item["tgt_image"].height)
                tgt_global_img_idx = 1 + len(crops_ref)
                first_crop_tgt_idx = tgt_global_img_idx + 1
            else:
                crops_tgt = []
                tgt_global_img_idx = 1
                first_crop_tgt_idx = 1
            crops_info.append((crops_tgt, first_crop_tgt_idx, tgt_global_img_idx))

        # Single forward pass for the whole batch
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        batch_sample_data = [[] for _ in range(B)]

        for layer in range(N_LAYERS):
            layer_hs = outputs.hidden_states[layer]  # [B, seq_len, D]

            for b_idx in range(B):
                sample_hs        = layer_hs[b_idx : b_idx + 1]          # [1, seq_len, D]
                sample_input_ids = inputs.input_ids[b_idx : b_idx + 1]  # [1, seq_len]
                crops_tgt, first_crop_tgt_idx, tgt_global_img_idx = crops_info[b_idx]

                all_token_values = []  # (token_str, prob) per bbox — stored in all_data
                all_tokens = []        # token_str only — used for Jaccard
                for k in range(4):
                    tv = get_decoded_tokens(
                        meta_batch[b_idx], k, sample_input_ids, sample_hs,
                        crops_tgt, first_crop_tgt_idx, tgt_global_img_idx
                    )
                    all_token_values.append(tv)
                    all_tokens.append([x[0] for x in tv])

                batch_sample_data[b_idx].append(all_token_values)

                filtered = [filter_tokens(all_tokens[i]) for i in range(4)]
                jac_scores = [
                    jaccard_similarity(filtered[i], filtered[j])
                    for i in range(4) for j in range(i + 1, 4)
                ]
                # print(np.max(jac_scores))
                per_layer_jac_scores[layer].extend(jac_scores)

        all_data.extend(batch_sample_data)

    return per_layer_jac_scores, all_data


per_layer_jac_scores, all_data = get_logit_metrics(
    model, test_dataset, num_samples=args.sample_size, batch_size=args.batch_size
)

layer_means = [
    [layer, np.mean(per_layer_jac_scores[layer])]
    for layer in sorted(per_layer_jac_scores)
]

print("\nMean Jaccard similarity per layer:")
for layer, score in layer_means:
    print(f"Layer {layer}: {score:.4f}")

layers, scores = zip(*layer_means)
plt.plot(layers, scores)
plt.xlabel("Layer")
plt.ylabel("Mean Jaccard Similarity")
plt.xlim(0, N_LAYERS)
plt.ylim(0, 1)
max_score = max(scores)
max_score_layer = scores.index(max_score)
plt.axhline(
    y=max_score,
    alpha=0.5,
    color='red',
    linestyle='--',
    label=f'Max Jaccard: {max_score:.4f} at Layer {max_score_layer}',
)
plt.legend()
# plt.show()

with open(f"{OUTPUT_FILE_NAME}_all_data.pkl", "wb") as f:
    pickle.dump(all_data, f)

with open(f"{OUTPUT_FILE_NAME}_jaccard.pkl", "wb") as f:
    pickle.dump(list(scores), f)

plt.savefig(f"{OUTPUT_FILE_NAME}.png")
