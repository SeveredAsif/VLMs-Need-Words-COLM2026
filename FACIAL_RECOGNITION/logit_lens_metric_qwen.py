import os
import pickle
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from data_class import VisionLanguageDataset
from jaccard_util import mean_jaccard_per_layer, pairwise_jaccard_for_bboxes
from plot_util import configure_plot_style, save_plot

BATCH_SIZE = 16
IGNORE_COLORS = False
IGNORE_OPTIONS = False
MODEL_SIZE = "8B"
UNKNOWN_DATASET_NAME = "FLUXSynID_EAST_ASIAN_DATASET"
KNOWN_DATASET_NAME = f"Qwen{MODEL_SIZE}_FAMOUS_ASIAN_DATASET"
SAMPLE_SIZE = -1

VISION_START_ID = 151652
VISION_END_ID = 151653
PATCH_SIZE = 16
MERGE_SIZE = 2
EFFECTIVE_BLOCK_SIZE = PATCH_SIZE * MERGE_SIZE
TGT_IMAGE_INDEX = 1

CHINESE_COLORS = frozenset(
    {
        "红",
        "蓝",
        "绿",
        "黄",
        "白",
        "黑",
        "橙",
        "紫",
        "粉",
        "灰",
        "棕",
        "青",
        "金",
        "银",
        "橘",
        "红色",
        "蓝色",
        "绿色",
        "黄色",
        "白色",
        "黑色",
        "橙色",
        "紫色",
        "粉色",
        "灰色",
        "棕色",
        "青色",
        "金色",
        "银色",
        "橘色",
        "粉红",
        "粉红色",
        "天蓝",
        "天蓝色",
        "深蓝",
        "浅蓝",
        "深红",
        "深绿",
        "浅绿",
        "橘红",
        "玫瑰",
        "玫瑰色",
    }
)


def find_qwen3vl_image_tokens(
    input_ids,
    image_resolution,
    bbox,
    image_index=0,
    vision_start_id=VISION_START_ID,
    vision_end_id=VISION_END_ID,
):
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()

    start_indices = [i for i, t in enumerate(input_ids) if t == vision_start_id]
    if image_index >= len(start_indices):
        raise ValueError(
            f"image_index {image_index} out of range; found {len(start_indices)} images."
        )

    global_start_idx = start_indices[image_index] + 1
    global_end_idx = input_ids.index(vision_end_id, global_start_idx)
    num_actual_tokens = global_end_idx - global_start_idx

    height, width = image_resolution
    grid_h = height // EFFECTIVE_BLOCK_SIZE
    grid_w = width // EFFECTIVE_BLOCK_SIZE

    if grid_h * grid_w != num_actual_tokens:
        raise ValueError(
            f"Resolution mismatch: {height}x{width} -> {grid_h}x{grid_w}={grid_h * grid_w} tokens, "
            f"but found {num_actual_tokens} between vision markers."
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
            px_start = gx * EFFECTIVE_BLOCK_SIZE
            px_end = (gx + 1) * EFFECTIVE_BLOCK_SIZE
            py_start = gy * EFFECTIVE_BLOCK_SIZE
            py_end = (gy + 1) * EFFECTIVE_BLOCK_SIZE
            if px_start < x_max and px_end > x_min and py_start < y_max and py_end > y_min:
                token_indices.append(global_start_idx + gy * grid_w + gx)

    return sorted(token_indices)


def get_processed_resolution(inputs, image_index):
    grid_thw = inputs.image_grid_thw[image_index]
    patch_h = int(grid_thw[1])
    patch_w = int(grid_thw[2])
    return patch_h * PATCH_SIZE, patch_w * PATCH_SIZE


def scale_bbox(original_box, image_obj, processed_h, processed_w):
    scale_x = processed_w / image_obj.width
    scale_y = processed_h / image_obj.height
    return (
        int(original_box[0][0] * scale_x),
        int(original_box[0][1] * scale_y),
        int(original_box[1][0] * scale_x),
        int(original_box[1][1] * scale_y),
    )


def get_decoded_tokens(
    model,
    processor,
    item,
    k,
    sample_input_ids,
    sample_hidden_state,
    proc_h_tgt,
    proc_w_tgt,
):
    original_box = item["tgt_coordinate"][k]
    tgt_image = item["tgt_image"]

    scaled_box = scale_bbox(original_box, tgt_image, proc_h_tgt, proc_w_tgt)
    target_indices = find_qwen3vl_image_tokens(
        input_ids=sample_input_ids,
        image_resolution=(proc_h_tgt, proc_w_tgt),
        bbox=scaled_box,
        image_index=TGT_IMAGE_INDEX,
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
        image_inputs, video_inputs = process_vision_info(messages_list)
        texts = processor.apply_chat_template(
            messages_list, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
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
    n_layers = len(model.model.language_model.layers) + 1

    for inputs, meta_batch in tqdm(dataloader):
        inputs = inputs.to(model.device)
        batch_size_actual = len(meta_batch)

        proc_resolutions = [
            get_processed_resolution(inputs, image_index=2 * b_idx + 1)
            for b_idx in range(batch_size_actual)
        ]

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        batch_sample_data = [[] for _ in range(batch_size_actual)]

        for layer in range(n_layers):
            layer_hs = outputs.hidden_states[layer]

            for b_idx in range(batch_size_actual):
                sample_hs = layer_hs[b_idx : b_idx + 1]
                sample_input_ids = inputs.input_ids[b_idx]
                proc_h_tgt, proc_w_tgt = proc_resolutions[b_idx]

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
                        proc_h_tgt,
                        proc_w_tgt,
                    )
                    all_token_values.append(tv)
                    all_tokens.append([x[0] for x in tv])

                batch_sample_data[b_idx].append(all_token_values)

                jac_scores = pairwise_jaccard_for_bboxes(
                    all_tokens,
                    ignore_colors=IGNORE_COLORS,
                    ignore_options=IGNORE_OPTIONS,
                    extra_color_tokens=CHINESE_COLORS,
                )
                per_layer_jac_scores[layer].extend(jac_scores)

        all_data.extend(batch_sample_data)

    return per_layer_jac_scores, all_data


def main():
    model_path = f"Qwen/Qwen3-VL-{MODEL_SIZE}-Instruct"

    unknown_dataset = VisionLanguageDataset(dataset_name=UNKNOWN_DATASET_NAME)
    known_dataset = VisionLanguageDataset(dataset_name=KNOWN_DATASET_NAME)

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
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

    with open(f"qwen_{MODEL_SIZE}_unknown_all_data.pkl", "wb") as f:
        pickle.dump(unknown_all_data, f)
    with open(f"qwen_{MODEL_SIZE}_known_all_data.pkl", "wb") as f:
        pickle.dump(known_all_data, f)

    save_plot(
        unknown,
        known,
        "AI Generated/Unknown",
        "Celebrity/Known",
        f"qwen_{MODEL_SIZE}_logit_jaccard.pdf",
    )


if __name__ == "__main__":
    main()
