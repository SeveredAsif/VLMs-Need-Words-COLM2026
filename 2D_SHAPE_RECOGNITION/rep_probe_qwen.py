# %%
import argparse
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
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from data_class import VisionLanguageDataset

print(f"Using model: {args.model_path}")

OUTPUT_FILE_NAME = f"rep_probe_results/QWEN/dataset{args.dataset}_model{args.model_path.replace('/', '_')}"
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
    Finds the global token indices in input_ids corresponding to a rectangular
    region in an image processed by Qwen3-VL.

    Args:
        input_ids: list[int] or 1-D torch.Tensor of token IDs.
        image_resolution: (height, width) of the image *as processed by the model*
                          (i.e. patch_h * 16, patch_w * 16).
        bbox: (x1, y1, x2, y2) in processed-image pixel coords.
        image_index: which image in the prompt (0-indexed).
        vision_start_id: token ID for <|vision_start|> (default 151652).
        vision_end_id: token ID for <|vision_end|> (default 151653).

    Returns:
        list[int]: sorted absolute token positions in input_ids.
    """
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()

    start_indices = [i for i, t in enumerate(input_ids) if t == vision_start_id]
    if image_index >= len(start_indices):
        raise ValueError(f"image_index {image_index} out of range; found {len(start_indices)} images.")

    global_start_idx = start_indices[image_index] + 1  # skip <|vision_start|>

    try:
        global_end_idx = input_ids.index(vision_end_id, global_start_idx)
    except ValueError:
        raise ValueError("Could not find matching <|vision_end|> token.")

    num_actual_tokens = global_end_idx - global_start_idx

    PATCH_SIZE = 16
    MERGE_SIZE = 2
    EFFECTIVE_BLOCK_SIZE = PATCH_SIZE * MERGE_SIZE  # 32

    height, width = image_resolution
    grid_h = height // EFFECTIVE_BLOCK_SIZE
    grid_w = width // EFFECTIVE_BLOCK_SIZE

    if grid_h * grid_w != num_actual_tokens:
        raise ValueError(
            f"Resolution mismatch: {height}x{width} implies {grid_h}x{grid_w}="
            f"{grid_h*grid_w} tokens, but found {num_actual_tokens} in input_ids."
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
            px_start, px_end = gx * EFFECTIVE_BLOCK_SIZE, (gx + 1) * EFFECTIVE_BLOCK_SIZE
            py_start, py_end = gy * EFFECTIVE_BLOCK_SIZE, (gy + 1) * EFFECTIVE_BLOCK_SIZE
            if px_start < x_max and px_end > x_min and py_start < y_max and py_end > y_min:
                local_idx = gy * grid_w + gx
                token_indices.append(global_start_idx + local_idx)

    return sorted(token_indices)


def get_processed_resolution(inputs, image_index):
    """Return (processed_h, processed_w) for the given image index."""
    grid_thw = inputs.image_grid_thw[image_index]
    patch_h = int(grid_thw[1])
    patch_w = int(grid_thw[2])
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
    inputs = processor(
        text=[text_input],
        images=[image1, image2],
        padding=True,
        return_tensors="pt",
    )

    # Get all hidden states ONCE per sample
    with torch.no_grad():
        outputs = model(**inputs.to(model.device), output_hidden_states=True)
        all_hidden_states = outputs.hidden_states

    answer_key = ord(test_dataset[k]["answer"]) - ord("A")

    # Precompute processed resolutions (stable across layers)
    proc_h_tgt, proc_w_tgt = get_processed_resolution(inputs, image_index=1)
    proc_h_ref, proc_w_ref = get_processed_resolution(inputs, image_index=0)

    for layer in range(N_LAYERS):
        layer_hidden_state = all_hidden_states[layer]

        target_representations = []
        for original_box in test_dataset[k]["tgt_positions"]:
            scaled_box = scale_bbox(original_box, image2, proc_h_tgt, proc_w_tgt)
            target_indices = find_qwen3vl_image_tokens(
                input_ids=inputs.input_ids[0],
                image_resolution=(proc_h_tgt, proc_w_tgt),
                bbox=scaled_box,
                image_index=1,
            )
            region_features = layer_hidden_state[0, target_indices, :]
            target_representations.append(region_features)

        reference_representations = []
        for original_box in test_dataset[k]["ref_positions"]:
            scaled_box = scale_bbox(original_box, image1, proc_h_ref, proc_w_ref)
            ref_indices = find_qwen3vl_image_tokens(
                input_ids=inputs.input_ids[0],
                image_resolution=(proc_h_ref, proc_w_ref),
                bbox=scaled_box,
                image_index=0,
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