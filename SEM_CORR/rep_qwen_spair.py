import argparse
import os
import pickle
import random

parser = argparse.ArgumentParser()
parser.add_argument('--box_size', type=int, required=True, help='Box size around keypoint')
parser.add_argument('--filters', nargs='+', required=True, help='List of filters')
parser.add_argument('--model_size', type=str, required=True, help='Model size (e.g. 2B, 4B, 8B)')
parser.add_argument('--cuda', type=int, required=True, help='CUDA device index')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from torch.utils.data import Subset
from data_classes import VisionLanguageDataset
from qwen_vl_utils import process_vision_info
from PIL import Image
import matplotlib.pyplot as plt


# model_path = f"Qwen/Qwen3-VL-{args.model_size}-Instruct"
model_path = "FINAL_SQUIGGLE_EXPERIMENT/qwen_2b_finetuned_on_task/best_model"

test_dataset = VisionLanguageDataset(split="test", box_size=args.box_size, filter_conditions=args.filters)
print(f"Using filter condition: {str(args.filters)} and box size: {args.box_size}")
print(f"Test dataset size: {len(test_dataset)}")

RESULTS_DIR = "spair_rep_results/QWEN_trained"
OUTPUT_FILE = f"{RESULTS_DIR}/correct_per_layer_box{args.box_size}_filters{str(args.filters)}_msize{args.model_size}"
os.makedirs(RESULTS_DIR, exist_ok=True)

SUBSAMPLE_TEST_SIZE = 1000
SEED = 42

random.seed(SEED)
num_samples = min(SUBSAMPLE_TEST_SIZE, len(test_dataset))
indices = random.sample(range(len(test_dataset)), num_samples)
test_dataset = Subset(test_dataset, indices)

# Load model and processor
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path, dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True
).cuda()

model.eval()

N_LAYERS = len(model.model.language_model.layers) + 1

print(f"Number of layers: {N_LAYERS}")

# Qwen3-VL image processing constants:
#   - 16px ViT patches with 2x2 spatial merge
#   - Each token covers 32x32 pixels
PATCH_SIZE = 16
MERGE_SIZE = 2
EFFECTIVE_BLOCK_SIZE = PATCH_SIZE * MERGE_SIZE  # 32

VISION_START_ID = 151652
VISION_END_ID = 151653


# -- Helpers -------------------------------------------------------------------

def find_qwen3vl_image_tokens(input_ids, image_resolution, bbox, image_index=0):
    """
    Map a bounding box to absolute token positions in Qwen3-VL's input_ids.

    Args:
        input_ids        : 1-D token ID list / tensor (no batch dim).
        image_resolution : (height, width) after processor resizing.
        bbox             : (x_min, y_min, x_max, y_max) in processed-image pixels.
        image_index      : 0 = first image, 1 = second image.

    Returns:
        Sorted list of absolute positions in input_ids.
    """
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()

    start_indices = [i for i, t in enumerate(input_ids) if t == VISION_START_ID]
    if image_index >= len(start_indices):
        raise ValueError(f"Image index {image_index} out of range. Found {len(start_indices)} images.")

    global_start_idx = start_indices[image_index] + 1

    try:
        global_end_idx = input_ids.index(VISION_END_ID, global_start_idx)
    except ValueError:
        raise ValueError("Could not find matching <|vision_end|> token.")

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

    tokens = []
    for gy in range(start_gy, end_gy):
        for gx in range(start_gx, end_gx):
            px_start, px_end = gx * EFFECTIVE_BLOCK_SIZE, (gx + 1) * EFFECTIVE_BLOCK_SIZE
            py_start, py_end = gy * EFFECTIVE_BLOCK_SIZE, (gy + 1) * EFFECTIVE_BLOCK_SIZE
            if (px_start < x_max and px_end > x_min and
                py_start < y_max and py_end > y_min):
                tokens.append(global_start_idx + gy * grid_w + gx)

    return sorted(tokens)


def _get_scaled_box(box, image_obj, grid_thw):
    """Scale a box from upscaled-image coords to processor-resized coords."""
    patch_h, patch_w = int(grid_thw[1]), int(grid_thw[2])
    processed_h, processed_w = patch_h * 16, patch_w * 16
    scale_x = processed_w / image_obj.width
    scale_y = processed_h / image_obj.height
    scaled = (
        int(box[0][0] * scale_x), int(box[0][1] * scale_y),
        int(box[1][0] * scale_x), int(box[1][1] * scale_y),
    )
    return scaled, (processed_h, processed_w)


def max_pool_cos_sim(ref_features, tgt_features):
    ref_pooled = torch.max(ref_features, dim=0).values
    tgt_pooled = torch.max(tgt_features, dim=0).values
    return torch.nn.functional.cosine_similarity(ref_pooled, tgt_pooled, dim=0).item()


def max_sim(ref_features, tgt_features):
    """ColBERT MaxSim: mean of per-ref-token maximum cosine similarity."""
    ref_norm = torch.nn.functional.normalize(ref_features, p=2, dim=1)
    tgt_norm = torch.nn.functional.normalize(tgt_features, p=2, dim=1)
    sim_matrix = torch.matmul(ref_norm, tgt_norm.T)
    return torch.mean(torch.max(sim_matrix, dim=1).values).item()


# -- Main evaluation loop -----------------------------------------------------

correct_per_layer = [0] * N_LAYERS
n_dataset = len(test_dataset)

for k in tqdm(range(n_dataset)):
    image1 = Image.open("SPair-71k/" + test_dataset[k]["og_src_img_path"]).convert("RGB")
    image2 = Image.open("SPair-71k/" + test_dataset[k]["og_tgt_img_path"]).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image1},
                {"type": "image", "image": image2},
                {"type": "text", "text": test_dataset[k]["prompt"]},
            ],
        }
    ]

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text_input],
        images=[image1, image2],
        padding=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs.to(model.device), output_hidden_states=True)
        all_hidden_states = outputs.hidden_states

    for layer in range(N_LAYERS):
        hs = all_hidden_states[layer]

        # ---- target box ----
        scaled_box, resolution = _get_scaled_box(test_dataset[k]["tgt_box"], image2, inputs.image_grid_thw[1])
        tgt_idx = find_qwen3vl_image_tokens(
            inputs.input_ids[0], image_resolution=resolution, bbox=scaled_box, image_index=1
        )
        tgt_rep = hs[0, tgt_idx, :]

        # ---- distractor boxes ----
        other_reps = []
        for ob in test_dataset[k]["other_boxes"]:
            scaled_box, resolution = _get_scaled_box(ob, image2, inputs.image_grid_thw[1])
            ob_idx = find_qwen3vl_image_tokens(
                inputs.input_ids[0], image_resolution=resolution, bbox=scaled_box, image_index=1
            )
            other_reps.append(hs[0, ob_idx, :])

        # ---- reference box ----
        scaled_box, resolution = _get_scaled_box(test_dataset[k]["ref_box"], image1, inputs.image_grid_thw[0])
        ref_idx = find_qwen3vl_image_tokens(
            inputs.input_ids[0], image_resolution=resolution, bbox=scaled_box, image_index=0
        )
        ref_rep = hs[0, ref_idx, :]

        # ---- score ----
        tgt_score = max_sim(ref_rep, tgt_rep)
        other_scores = [max_sim(ref_rep, o) for o in other_reps]
        if tgt_score > max(other_scores):
            correct_per_layer[layer] += 1

    torch.cuda.empty_cache()


# -- Results -------------------------------------------------------------------

print("\nResults per layer:")
for layer in range(N_LAYERS):
    acc = correct_per_layer[layer] / n_dataset
    print(f"Layer {layer}: {acc:.4f} ({correct_per_layer[layer]}/{n_dataset})")

# -- Plot ----------------------------------------------------------------------

accuracies = [correct_per_layer[l] / n_dataset for l in range(N_LAYERS)]

plt.figure()
plt.plot(range(N_LAYERS), accuracies)

max_acc = max(accuracies)
max_layer = accuracies.index(max_acc)
plt.axhline(y=max_acc, alpha=0.5, color='red', linestyle='--',
            label=f'Max Accuracy: {max_acc:.4f} at Layer {max_layer}')

plt.legend()
plt.xlabel("Layer")
plt.ylabel("Accuracy")
plt.xlim(0, N_LAYERS)
plt.ylim(0, 1)

with open(OUTPUT_FILE + ".pkl", "wb") as f:
    pickle.dump(correct_per_layer, f)

plt.savefig(OUTPUT_FILE + ".png")
