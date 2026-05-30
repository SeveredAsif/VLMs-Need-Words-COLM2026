import argparse
import os
import math
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
from transformers import AutoModelForImageTextToText, AutoProcessor
from tqdm import tqdm
from torch.utils.data import Subset
from data_classes import VisionLanguageDataset
from PIL import Image
import matplotlib.pyplot as plt


model_path = f"OpenGVLab/InternVL3_5-{args.model_size}-HF"

test_dataset = VisionLanguageDataset(split="test", box_size=args.box_size, filter_conditions=args.filters)
print(f"Using filter condition: {str(args.filters)} and box size: {args.box_size}")
print(f"Test dataset size: {len(test_dataset)}")

RESULTS_DIR = "spair_rep_results/INTERNVL3_5"
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
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
).cuda()

model.eval()

N_LAYERS = len(model.language_model.layers) + 1

print(f"Number of layers: {N_LAYERS}")

# InternVL3 image processing constants:
#   - 448×448 image tile
#   - 14px ViT patches → 32×32 patches per tile
#   - 2×2 pixel shuffle → 16×16 tokens per tile
#   - Each token covers 28×28 pixels (448/16)
INTERNVL_TILE_SIZE = 448
TOKENS_PER_TILE_SIDE = 16
PX_PER_TOKEN = INTERNVL_TILE_SIZE // TOKENS_PER_TILE_SIDE  # 28

img_context_token_id = processor.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
print(f"Image context token ID: {img_context_token_id}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_consecutive_groups(positions):
    """Group a sorted list of ints into runs of consecutive values."""
    if not positions:
        return []
    groups, current = [], [positions[0]]
    for p in positions[1:]:
        if p == current[-1] + 1:
            current.append(p)
        else:
            groups.append(current)
            current = [p]
    groups.append(current)
    return groups


def infer_spatial_tile_layout(n_spatial_tiles, img_w, img_h):
    """
    Given the number of spatial tiles and the image aspect ratio, pick (tile_h, tile_w)
    whose ratio best matches img_w / img_h (InternVL3 selects tiles this way).
    """
    if n_spatial_tiles == 1:
        return 1, 1
    aspect = img_w / img_h
    best, best_diff = (1, n_spatial_tiles), float('inf')
    for th in range(1, n_spatial_tiles + 1):
        if n_spatial_tiles % th == 0:
            tw = n_spatial_tiles // th
            diff = abs(tw / th - aspect)
            if diff < best_diff:
                best_diff = diff
                best = (th, tw)
    return best  # (tile_h, tile_w)


def find_internvl3_image_tokens(input_ids, bbox, image_index,
                                img_context_token_id,
                                img_w, img_h):
    """
    Map a bounding box (in original image pixel coordinates) to absolute
    token positions in InternVL3-hf's input_ids.

    Args:
        input_ids        : 1-D token ID list / tensor (no batch dim).
        bbox             : (x_min, y_min, x_max, y_max) in image pixels.
        image_index      : 0 = first image, 1 = second image.
        img_context_token_id : token ID of <IMG_CONTEXT>.
        img_w/h          : dimensions of the image fed to the processor.

    Returns:
        Sorted list of absolute positions in input_ids.
    """
    ids = input_ids.squeeze().tolist()

    img_positions = [i for i, t in enumerate(ids) if t == img_context_token_id]
    groups = find_consecutive_groups(img_positions)
    group = groups[image_index]
    total_tokens = len(group)
    global_start = group[0]

    n_tiles_total = total_tokens // TOKENS_PER_TILE_SIDE ** 2  # total 256-token blocks

    if n_tiles_total <= 1:
        # Single tile — no thumbnail
        canvas_h = canvas_w = INTERNVL_TILE_SIZE  # 448×448
        tile_w = 1
        token_start = global_start
    else:
        # Spatial tiles first, thumbnail last (InternVL3-hf ordering)
        n_spatial = n_tiles_total - 1
        tile_h, tile_w = infer_spatial_tile_layout(n_spatial, img_w, img_h)
        canvas_h = tile_h * INTERNVL_TILE_SIZE
        canvas_w = tile_w * INTERNVL_TILE_SIZE
        token_start = global_start  # spatial tiles are first; thumbnail tokens are at the end

    # Scale bbox from image coords to spatial-tile canvas coords
    scale_x = canvas_w / img_w
    scale_y = canvas_h / img_h
    x_min, y_min, x_max, y_max = bbox

    grid_h = (canvas_h // INTERNVL_TILE_SIZE) * TOKENS_PER_TILE_SIDE
    grid_w = (canvas_w // INTERNVL_TILE_SIZE) * TOKENS_PER_TILE_SIDE

    start_gx = max(0, int(x_min * scale_x / PX_PER_TOKEN))
    end_gx   = min(grid_w, math.ceil(x_max * scale_x / PX_PER_TOKEN))
    start_gy = max(0, int(y_min * scale_y / PX_PER_TOKEN))
    end_gy   = min(grid_h, math.ceil(y_max * scale_y / PX_PER_TOKEN))

    # InternVL3 stores tokens tile-by-tile: all 256 tokens of tile 0, then tile 1, etc.
    # NOT flat row-major across the whole canvas.
    tokens = []
    for gy in range(start_gy, end_gy):
        for gx in range(start_gx, end_gx):
            t_row = gy // TOKENS_PER_TILE_SIDE
            t_col = gx // TOKENS_PER_TILE_SIDE
            t_idx = t_row * tile_w + t_col
            l_gy  = gy % TOKENS_PER_TILE_SIDE
            l_gx  = gx % TOKENS_PER_TILE_SIDE
            rel   = t_idx * TOKENS_PER_TILE_SIDE**2 + l_gy * TOKENS_PER_TILE_SIDE + l_gx
            tokens.append(token_start + rel)

    # print(f"  bbox={bbox} → {len(tokens)} tokens")
    return tokens


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


# ── Main evaluation loop ──────────────────────────────────────────────────────

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
    # InternVL3-hf processor expects list-of-image-lists and list-of-strings
    inputs = processor([[image1, image2]], [text_input], padding=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs.to(model.device), output_hidden_states=True)
        all_hidden_states = outputs.hidden_states  # tuple of N_LAYERS tensors

    for layer in range(N_LAYERS):
        hs = all_hidden_states[layer]  # [1, seq_len, hidden_dim]

        # ---- target box ----
        tgt_box = test_dataset[k]["tgt_box"]
        tgt_idx = find_internvl3_image_tokens(
            inputs.input_ids[0],
            bbox=(tgt_box[0][0], tgt_box[0][1], tgt_box[1][0], tgt_box[1][1]),
            image_index=1,
            img_context_token_id=img_context_token_id,
            img_w=image2.width,
            img_h=image2.height,
        )
        tgt_rep = hs[0, tgt_idx, :]

        # ---- distractor boxes ----
        other_reps = []
        for ob in test_dataset[k]["other_boxes"]:
            ob_idx = find_internvl3_image_tokens(
                inputs.input_ids[0],
                bbox=(ob[0][0], ob[0][1], ob[1][0], ob[1][1]),
                image_index=1,
                img_context_token_id=img_context_token_id,
                img_w=image2.width,
                img_h=image2.height,
            )
            other_reps.append(hs[0, ob_idx, :])

        # ---- reference box ----
        ref_box = test_dataset[k]["ref_box"]
        ref_idx = find_internvl3_image_tokens(
            inputs.input_ids[0],
            bbox=(ref_box[0][0], ref_box[0][1], ref_box[1][0], ref_box[1][1]),
            image_index=0,
            img_context_token_id=img_context_token_id,
            img_w=image1.width,
            img_h=image1.height,
        )
        ref_rep = hs[0, ref_idx, :]

        # ---- score ----
        tgt_score    = max_sim(ref_rep, tgt_rep)
        other_scores = [max_sim(ref_rep, o) for o in other_reps]
        if tgt_score > max(other_scores):
            correct_per_layer[layer] += 1

    torch.cuda.empty_cache()


# ── Results ───────────────────────────────────────────────────────────────────

print("\nResults per layer:")
for layer in range(N_LAYERS):
    acc = correct_per_layer[layer] / n_dataset
    print(f"Layer {layer}: {acc:.4f} ({correct_per_layer[layer]}/{n_dataset})")

# ── Plot ──────────────────────────────────────────────────────────────────────

accuracies = [correct_per_layer[l] / n_dataset for l in range(N_LAYERS)]

plt.figure()
plt.plot(range(N_LAYERS), accuracies)

max_acc   = max(accuracies)
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
