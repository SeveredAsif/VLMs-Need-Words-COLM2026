# %%
import argparse
import os
import pickle
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
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
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from data_class import VisionLanguageDataset

IGNORE_COLORS = args.ignore_colors
IGNORE_OPTIONS = args.ignore_options

print(f"Using model: {args.model_path}")

OUTPUT_FILE_NAME = f"logit_lens_results/QWEN/dataset{args.dataset}_model{args.model_path.replace('/', '_')}"
os.makedirs(os.path.dirname(OUTPUT_FILE_NAME), exist_ok=True)

test_dataset = VisionLanguageDataset(dataset=args.dataset)

processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left')

model = Qwen3VLForConditionalGeneration.from_pretrained(
    args.model_path, dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True
).cuda()

model.eval()

N_LAYERS = len(model.model.language_model.layers) + 1


def find_qwen3vl_image_tokens(input_ids, image_resolution, bbox, image_index=0,
                               vision_start_id=151652, vision_end_id=151653):
    """
    Finds the absolute token positions in input_ids corresponding to a
    rectangular region in a Qwen3-VL processed image.
    """
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()

    start_indices = [i for i, t in enumerate(input_ids) if t == vision_start_id]
    if image_index >= len(start_indices):
        raise ValueError(f"image_index {image_index} out of range; found {len(start_indices)} images.")

    global_start_idx = start_indices[image_index] + 1  # skip <|vision_start|>
    global_end_idx   = input_ids.index(vision_end_id, global_start_idx)
    num_actual_tokens = global_end_idx - global_start_idx

    PATCH_SIZE = 16
    MERGE_SIZE = 2
    EFFECTIVE_BLOCK_SIZE = PATCH_SIZE * MERGE_SIZE  # 32 px per token

    height, width = image_resolution
    grid_h = height // EFFECTIVE_BLOCK_SIZE
    grid_w = width  // EFFECTIVE_BLOCK_SIZE

    if grid_h * grid_w != num_actual_tokens:
        raise ValueError(
            f"Resolution mismatch: {height}x{width} -> {grid_h}x{grid_w}={grid_h*grid_w} tokens, "
            f"but found {num_actual_tokens} between vision markers."
        )

    x1, y1, x2, y2 = bbox
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    start_gx = max(0, int(x_min / EFFECTIVE_BLOCK_SIZE))
    end_gx   = min(grid_w, int((x_max + EFFECTIVE_BLOCK_SIZE - 1) / EFFECTIVE_BLOCK_SIZE))
    start_gy = max(0, int(y_min / EFFECTIVE_BLOCK_SIZE))
    end_gy   = min(grid_h, int((y_max + EFFECTIVE_BLOCK_SIZE - 1) / EFFECTIVE_BLOCK_SIZE))

    token_indices = []
    for gy in range(start_gy, end_gy):
        for gx in range(start_gx, end_gx):
            px_start = gx * EFFECTIVE_BLOCK_SIZE;  px_end = (gx + 1) * EFFECTIVE_BLOCK_SIZE
            py_start = gy * EFFECTIVE_BLOCK_SIZE;  py_end = (gy + 1) * EFFECTIVE_BLOCK_SIZE
            if px_start < x_max and px_end > x_min and py_start < y_max and py_end > y_min:
                token_indices.append(global_start_idx + gy * grid_w + gx)

    return sorted(token_indices)


def get_processed_resolution(inputs, image_index):
    """Return (processed_h, processed_w) for the given image index from image_grid_thw."""
    grid_thw = inputs.image_grid_thw[image_index]
    patch_h  = int(grid_thw[1])
    patch_w  = int(grid_thw[2])
    return patch_h * 16, patch_w * 16


def scale_bbox(original_box, image_obj, processed_h, processed_w):
    """Scale a [[x1,y1],[x2,y2]] box from original pixel coords to processed-image coords."""
    scale_x = processed_w / image_obj.width
    scale_y = processed_h / image_obj.height
    return (
        int(original_box[0][0] * scale_x),
        int(original_box[0][1] * scale_y),
        int(original_box[1][0] * scale_x),
        int(original_box[1][1] * scale_y),
    )


def get_decoded_tokens(item, k, sample_input_ids, sample_hidden_state, proc_h_tgt, proc_w_tgt):
    """
    Project hidden states for bbox k of item's target image through the lm_head.
    Returns list of (token_str, prob) for the top-1 token at each spatial position.
    """
    original_box = item["tgt_positions"][k]
    tgt_image    = item["tgt_image"]

    scaled_box     = scale_bbox(original_box, tgt_image, proc_h_tgt, proc_w_tgt)
    target_indices = find_qwen3vl_image_tokens(
        input_ids=sample_input_ids,
        image_resolution=(proc_h_tgt, proc_w_tgt),
        bbox=scaled_box,
        image_index=1,  # tgt is always the second image
    )

    region_features = sample_hidden_state[0, target_indices, :]

    with torch.no_grad():
        normalized = model.model.language_model.norm(region_features)
        logits     = model.lm_head(normalized)
        probs      = torch.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=5, dim=-1)

    token_values = []
    for t in range(region_features.shape[0]):
        tok = processor.tokenizer.decode([top_ids[t, 0].item()])
        token_values.append((tok, top_probs[t, 0].item()))
    return token_values


def collate_fn(batch):
    messages_list = [[
        {"role": "user", "content": [
            {"type": "image", "image": item["ref_image"]},
            {"type": "image", "image": item["tgt_image"]},
            {"type": "text",  "text":  item["prompt"]},
        ]}
    ] for item in batch]

    image_inputs, video_inputs = process_vision_info(messages_list)
    texts  = processor.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt")
    return inputs, batch


# Chinese color words — single characters and common multi-character forms.
# Qwen's tokenizer may emit either, so we cover both.
CHINESE_COLORS = {
    # single-character roots
    '红', '蓝', '绿', '黄', '白', '黑', '橙', '紫', '粉', '灰',
    '棕', '青', '金', '银', '橘',
    # -色 variants
    '红色', '蓝色', '绿色', '黄色', '白色', '黑色', '橙色', '紫色',
    '粉色', '灰色', '棕色', '青色', '金色', '银色', '橘色',
    # compound colors
    '粉红', '粉红色', '天蓝', '天蓝色', '深蓝', '浅蓝', '深红', '深绿',
    '浅绿', '橘红', '玫瑰', '玫瑰色',
}


def is_color(name):
    try:
        webcolors.name_to_hex(name)
        return True
    except ValueError:
        return False


def is_chinese_color(token):
    return token in CHINESE_COLORS


def filter_tokens(tokens):
    """Drop color names and/or option letters based on IGNORE_COLORS / IGNORE_OPTIONS flags."""
    result = []
    for t in tokens:
        t_stripped = t.strip()
        if IGNORE_COLORS and (is_color(t_stripped.lower()) or is_chinese_color(t_stripped)): continue
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

        # Precompute processed resolutions for all images in the batch.
        # image_grid_thw is ordered [ref_0, tgt_0, ref_1, tgt_1, ...]
        proc_resolutions = [
            get_processed_resolution(inputs, image_index=2 * b_idx + 1)  # tgt
            for b_idx in range(B)
        ]

        # Single forward pass for the whole batch
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        batch_sample_data = [[] for _ in range(B)]

        for layer in range(N_LAYERS):
            layer_hs = outputs.hidden_states[layer]  # [B, seq_len, D]

            for b_idx in range(B):
                sample_hs        = layer_hs[b_idx : b_idx + 1]  # [1, seq_len, D]
                sample_input_ids = inputs.input_ids[b_idx]       # [seq_len] 1-D
                proc_h_tgt, proc_w_tgt = proc_resolutions[b_idx]

                all_token_values = []
                all_tokens = []
                for k in range(4):
                    tv = get_decoded_tokens(
                        meta_batch[b_idx], k, sample_input_ids, sample_hs,
                        proc_h_tgt, proc_w_tgt
                    )
                    all_token_values.append(tv)
                    all_tokens.append([x[0] for x in tv])

                batch_sample_data[b_idx].append(all_token_values)

                filtered = [filter_tokens(all_tokens[i]) for i in range(4)]
                jac_scores = [
                    jaccard_similarity(filtered[i], filtered[j])
                    for i in range(4) for j in range(i + 1, 4)
                ]
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
