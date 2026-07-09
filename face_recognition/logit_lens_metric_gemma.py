import math
import os
import pickle
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from data_class import VisionLanguageDataset
from jaccard_util import mean_jaccard_per_layer, pairwise_jaccard_for_bboxes
from plot_util import configure_plot_style, save_plot

BATCH_SIZE = 16
IGNORE_COLORS = False
IGNORE_OPTIONS = False
MSIZE = "12b"
KNOWN_DATASET_NAME = "Gemma12B_FAMOUS_EAST_ASIAN_DATASET"
UNKNOWN_DATASET_NAME = "FLUXSynID_EAST_ASIAN_DATASET"
SAMPLE_SIZE = -1
OUTPUT_DIR = f"logit_lens_results/GEMMA/{MSIZE}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


VISION_START_ID = 255999
RESIZED_SIZE = 896
TOKENS_PER_SIDE = 16


def compute_pan_and_scan_crops(
    width,
    height,
    min_crop_size=256,
    max_num_crops=4,
    min_ratio=1.2,
):
    """Replicate Gemma3ImageProcessor.pan_and_scan crop boundaries."""
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


def bbox_to_token_indices(x_min, y_min, x_max, y_max, original_width, original_height):
    width_ratio = RESIZED_SIZE / original_width
    height_ratio = RESIZED_SIZE / original_height
    patch_size = RESIZED_SIZE / TOKENS_PER_SIDE

    x_min_token = int(x_min * width_ratio // patch_size)
    x_max_token = max(int(x_max * width_ratio // patch_size), x_min_token + 1)
    y_min_token = int(y_min * height_ratio // patch_size)
    y_max_token = max(int(y_max * height_ratio // patch_size), y_min_token + 1)

    x_min_token = max(0, min(x_min_token, TOKENS_PER_SIDE - 1))
    y_min_token = max(0, min(y_min_token, TOKENS_PER_SIDE - 1))
    x_max_token = max(0, min(x_max_token, TOKENS_PER_SIDE))
    y_max_token = max(0, min(y_max_token, TOKENS_PER_SIDE))

    return [
        y * TOKENS_PER_SIDE + x
        for y in range(y_min_token, y_max_token)
        for x in range(x_min_token, x_max_token)
    ]


def get_absolute_token_positions(token_indices, input_ids, image_index=0):
    image_starts = [i for i, x in enumerate(input_ids[0].tolist()) if x == VISION_START_ID]
    offset = image_starts[image_index] + 1
    return [i + offset for i in token_indices]


def find_tokens_in_crops(input_ids, bbox, crops, first_crop_input_ids_idx):
    image_starts = [i for i, x in enumerate(input_ids[0].tolist()) if x == VISION_START_ID]
    patch_size = RESIZED_SIZE / TOKENS_PER_SIDE
    ox1, oy1, ox2, oy2 = bbox

    all_positions = []
    for crop_idx, (cx1, cy1, cx2, cy2) in enumerate(crops):
        ix1 = max(ox1, cx1)
        iy1 = max(oy1, cy1)
        ix2 = min(ox2, cx2)
        iy2 = min(oy2, cy2)
        if ix1 >= ix2 or iy1 >= iy2:
            continue

        crop_w = cx2 - cx1
        crop_h = cy2 - cy1
        scale_x = RESIZED_SIZE / crop_w
        scale_y = RESIZED_SIZE / crop_h

        lx1 = (ix1 - cx1) * scale_x
        ly1 = (iy1 - cy1) * scale_y
        lx2 = (ix2 - cx1) * scale_x
        ly2 = (iy2 - cy1) * scale_y

        tx1 = int(lx1 // patch_size)
        ty1 = int(ly1 // patch_size)
        tx2 = max(int(lx2 // patch_size), tx1 + 1)
        ty2 = max(int(ly2 // patch_size), ty1 + 1)

        tx1 = max(0, min(tx1, TOKENS_PER_SIDE - 1))
        ty1 = max(0, min(ty1, TOKENS_PER_SIDE - 1))
        tx2 = max(0, min(tx2, TOKENS_PER_SIDE))
        ty2 = max(0, min(ty2, TOKENS_PER_SIDE))

        token_indices = [
            y * TOKENS_PER_SIDE + x for y in range(ty1, ty2) for x in range(tx1, tx2)
        ]
        offset = image_starts[first_crop_input_ids_idx + crop_idx] + 1
        all_positions.extend(t + offset for t in token_indices)

    return all_positions


def get_decoded_tokens(
    model,
    processor,
    item,
    k,
    sample_input_ids,
    sample_hidden_state,
    crops_tgt,
    first_crop_tgt_idx,
    tgt_global_img_idx,
):
    original_box = item["tgt_coordinate"][k]
    bbox = (
        original_box[0][0],
        original_box[0][1],
        original_box[1][0],
        original_box[1][1],
    )
    tgt_image = item["tgt_image"]

    if crops_tgt:
        target_indices = find_tokens_in_crops(
            input_ids=sample_input_ids,
            bbox=bbox,
            crops=crops_tgt,
            first_crop_input_ids_idx=first_crop_tgt_idx,
        )
    else:
        rel = bbox_to_token_indices(
            bbox[0], bbox[1], bbox[2], bbox[3], tgt_image.width, tgt_image.height
        )
        target_indices = get_absolute_token_positions(
            rel, sample_input_ids, image_index=tgt_global_img_idx
        )

    region_features = sample_hidden_state[0, target_indices, :]

    with torch.no_grad():
        normalized = model.model.language_model.norm(region_features)
        logits = model.lm_head(normalized)
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=5, dim=-1)

    token_values = []
    for t in range(region_features.shape[0]):
        tok = processor.tokenizer.decode([top_ids[t, 0].item()]).strip()
        token_values.append((tok, top_probs[t, 0].item()))
    return token_values


def make_collate_fn(processor):
    def collate_fn(batch):
        images_list = [[item["ref_image"], item["tgt_image"]] for item in batch]
        messages_list = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["ref_image"]},
                        {"type": "image", "image": item["tgt_image"]},
                        {"type": "text", "text": item["prompt"]},
                    ],
                }
            ]
            for item in batch
        ]
        texts = processor.apply_chat_template(
            messages_list, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(images_list, texts, padding=True, return_tensors="pt")
        return inputs, batch

    return collate_fn


def get_logit_metrics(model, processor, target_dataset, num_samples=-1, batch_size=BATCH_SIZE):
    if num_samples == -1:
        num_samples = len(target_dataset)

    subset = Subset(target_dataset, range(num_samples))
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(processor),
    )

    per_layer_jac_scores = defaultdict(list)
    all_data = []

    for inputs, meta_batch in tqdm(dataloader):
        inputs = inputs.to(model.device)
        batch_size_actual = len(meta_batch)

        crops_info = []
        for item in meta_batch:
            crops_ref = compute_pan_and_scan_crops(
                item["ref_image"].width, item["ref_image"].height
            )
            crops_tgt = compute_pan_and_scan_crops(
                item["tgt_image"].width, item["tgt_image"].height
            )
            tgt_global_img_idx = 1 + len(crops_ref)
            first_crop_tgt_idx = tgt_global_img_idx + 1
            crops_info.append((crops_tgt, first_crop_tgt_idx, tgt_global_img_idx))

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        batch_sample_data = [[] for _ in range(batch_size_actual)]
        n_layers = len(model.model.language_model.layers)

        for layer in range(n_layers):
            layer_hs = outputs.hidden_states[layer]

            for b_idx in range(batch_size_actual):
                sample_hs = layer_hs[b_idx : b_idx + 1]
                sample_input_ids = inputs.input_ids[b_idx : b_idx + 1]
                crops_tgt, first_crop_tgt_idx, tgt_global_img_idx = crops_info[b_idx]

                all_token_values = []
                all_tokens = []
                for k in range(4):
                    tv = get_decoded_tokens(
                        model,
                        processor,
                        meta_batch[b_idx],
                        k,
                        sample_input_ids,
                        sample_hs,
                        crops_tgt,
                        first_crop_tgt_idx,
                        tgt_global_img_idx,
                    )
                    all_token_values.append(tv)
                    all_tokens.append([x[0] for x in tv])

                batch_sample_data[b_idx].append(all_token_values)

                jac_scores = pairwise_jaccard_for_bboxes(
                    all_tokens,
                    ignore_colors=IGNORE_COLORS,
                    ignore_options=IGNORE_OPTIONS,
                )
                per_layer_jac_scores[layer].extend(jac_scores)

        all_data.extend(batch_sample_data)

    return per_layer_jac_scores, all_data


def main():
    model_path = f"google/gemma-3-{MSIZE}-it"

    unknown_dataset = VisionLanguageDataset(dataset_name=UNKNOWN_DATASET_NAME)
    known_dataset = VisionLanguageDataset(dataset_name=KNOWN_DATASET_NAME)

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left", do_pan_and_scan=True
    )
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).cuda().eval()

    unknown_per_layer_jac_scores, unknown_all_data = get_logit_metrics(
        model, processor, unknown_dataset, num_samples=SAMPLE_SIZE
    )
    known_per_layer_jac_scores, known_all_data = get_logit_metrics(
        model, processor, known_dataset, num_samples=SAMPLE_SIZE
    )

    unknown = mean_jaccard_per_layer(unknown_per_layer_jac_scores)
    known = mean_jaccard_per_layer(known_per_layer_jac_scores)

    configure_plot_style()

    with open(f"{OUTPUT_DIR}/gemma_{MSIZE}_unknown_all_data.pkl", "wb") as f:
        pickle.dump(unknown_all_data, f)
    with open(f"{OUTPUT_DIR}/gemma_{MSIZE}_known_all_data.pkl", "wb") as f:
        pickle.dump(known_all_data, f)

    save_plot(
        unknown,
        known,
        "AI Generated/Unknown",
        "Celebrity/Known",
        f"{OUTPUT_DIR}/gemma_{MSIZE}_logit_jaccard.pdf",
    )


if __name__ == "__main__":
    main()
