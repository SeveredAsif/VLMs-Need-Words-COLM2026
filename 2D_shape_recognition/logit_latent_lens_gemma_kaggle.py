"""
Self-contained Kaggle notebook script: logit lens + LatentLens (McGill-NLP) probing
on the 2D_shape_recognition dataset (basic_shapes_TEST / squiggles_30_TEST / etc.),
using Gemma-3, sharded across a Kaggle T4x2 instance.

HOW TO USE ON KAGGLE
--------------------
1. Add this whole repo as a Kaggle "Dataset" input, e.g. at:
     /kaggle/input/datasets/aliasifkhan131/vlms-need-words/VLMs-Need-Words-COLM2026
   (adjust DATASET_ROOT below if your input slug differs)
2. Add the Gemma-3 model as a Kaggle "Model" input (or let from_pretrained pull
   from the Hub if internet is on for the session).
3. Set the Kaggle accelerator to "GPU T4 x2".
4. Paste ###CELL 1 into one notebook cell and run it. It installs `latentlens`,
   runs both lenses over the dataset, prints per-layer Jaccard-distinctiveness
   tables for each lens, and saves plots + pickles to /kaggle/working.
5. Paste ###CELL 2 into a separate cell and run it after Cell 1 (or after a
   kernel restart, since it only reads the saved pickles back -- no GPU needed).
   It reproduces logit_lens_traj.ipynb's per-patch trajectory plots, but works
   for EITHER lens via the LENS toggle at the top of the cell.

WHAT CHANGED FROM logit_lens_metric_gemma.py
---------------------------------------------
- No argparse / data_class.py import -- everything needed is inlined so the file
  can be pasted into a single Kaggle cell with nothing else on the input path.
- device_map="auto" (T4x2) instead of .cuda() (single GPU), since a 12B model
  does not fit comfortably on one T4.
- Added a second, parallel lens: for every (layer, option-bbox) pair that the
  logit lens already decodes via `model.lm_head`, LatentLens instead does a
  nearest-neighbor lookup in representation space against a corpus index built
  once from the LatentLens package's own bundled concepts.txt. Same Jaccard
  "distinctiveness across the 4 options" metric is computed for both lenses, so
  the two can be compared layer-for-layer.
- All bbox -> token-index math (compute_pan_and_scan_crops, bbox_to_token_indices,
  get_absolute_token_positions, find_tokens_in_crops) is copied verbatim from
  logit_lens_metric_gemma.py -- unchanged, since that logic is dataset/geometry
  specific and has nothing to do with which lens reads out the tokens.
"""

# %%
###CELL 1
# ----------------------------------------------------------------------------
# (Uncomment on first run in a fresh Kaggle session; safe to leave commented on
#  reruns once packages are installed.)
# !pip install -q -U latentlens accelerate webcolors
# ----------------------------------------------------------------------------

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import glob
import json
import math
import pickle
import random
import re
from collections import defaultdict
from types import SimpleNamespace

import torch
import numpy as np
import matplotlib.pyplot as plt
import webcolors
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from tqdm import tqdm

import latentlens
#https://www.kaggle.com/code/aliasifkhan131/latent-lens-sameen-bhai-ds?scriptVersionId=344298954
# ============================== CONFIG =======================================
args = SimpleNamespace(
    dataset_name="basic_shapes_TEST",       # folder name under DATASET_ROOT (matches VisionLanguageDataset's `dataset` arg)
    dataset_root="/home/user8/VLMs-Need-Words-COLM2026/2D_shape_recognition",
    model_path="google/gemma-3-12b-it",
    device_map="auto",                      # T4 x2 -- 12B model needs both GPUs
    dtype="bfloat16",                       # float16 overflows Gemma's later-layer activations -> NaN
    attn_implementation="sdpa",             # "flash_attention_2" only if you've installed it on Kaggle
    ps=True,                                # do pan-and-scan
    batch_size=8,
    sample_size=-1,                         # -1 = full dataset
    ignore_colors=True,
    ignore_options=True,
    ignore_special=True,   # drop <bos>/<eos>/... and pure-punctuation tokens from the Jaccard metric
    latent_skip_special=True,  # in get_latent_tokens, skip past <bos>-like neighbors to the next rank
    corpus=None,                            # None = auto-locate bundled concepts.txt inside installed latentlens package
    max_index_sentences=3000,               # subsample concepts.txt for a fast index build
    index_cache_dir="/home/user8/VLMs-Need-Words-COLM2026/latentlens_index",
    latentlens_top_k=20,  # bumped up from 5 -- gives get_latent_tokens more candidates to skip
                           # past <bos>/punctuation before falling back to it (see IGNORE_SPECIAL)
    index_batch_size=32,
    output_root="/home/user8/VLMs-Need-Words-COLM2026/2D_shape_recognition/logit_latent_results",
)

IGNORE_COLORS = args.ignore_colors
IGNORE_OPTIONS = args.ignore_options
IGNORE_SPECIAL = args.ignore_special
DO_PAN_AND_SCAN = args.ps

OUTPUT_FILE_NAME = f"{args.output_root}/dataset{args.dataset_name}_model{args.model_path.replace('/', '_')}"
os.makedirs(os.path.dirname(OUTPUT_FILE_NAME), exist_ok=True)


# ============================== DATASET ======================================
PROMPT = ('Which shape in the second image is most similar to the REF shape in the first image?\n'
          'Select from the following choices.\n(A) Point A\n(B) Point B\n(C) Point C\n(D) Point D\n')


class VisionLanguageDataset(Dataset):
    """Inlined copy of data_class.VisionLanguageDataset, with an added
    dataset_root prefix so it works when the repo is mounted as a Kaggle input
    instead of being the current working directory."""

    def __init__(self, dataset, dataset_root=""):
        answers_path = os.path.join(dataset_root, dataset, "answers.json")
        with open(answers_path, "r") as f:
            self.data = json.load(f)
        self.dataset_root = dataset_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        ref_image_path = os.path.join(self.dataset_root, item["ref_image_path"])
        tgt_image_path = os.path.join(self.dataset_root, item["tgt_image_path"])

        ref_image = Image.open(ref_image_path).convert("RGB")
        tgt_image = Image.open(tgt_image_path).convert("RGB")

        return {
            "prompt": PROMPT,
            "answer": item["answer"],
            "ref_image": ref_image,
            "tgt_image": tgt_image,
            "ref_positions": item["ref_pixel_positions"],
            "tgt_positions": item["tgt_pixel_positions"],
            "shape_types": item["shape_types"],
        }


test_dataset = VisionLanguageDataset(args.dataset_name, dataset_root=args.dataset_root)
print(f"Loaded {len(test_dataset)} items from {args.dataset_name}")


# ============================== MODEL ========================================
print(f"Using model: {args.model_path}")

processor = AutoProcessor.from_pretrained(
    args.model_path, trust_remote_code=True, padding_side='left', do_pan_and_scan=DO_PAN_AND_SCAN
)
model = Gemma3ForConditionalGeneration.from_pretrained(
    args.model_path,
    dtype=getattr(torch, args.dtype),
    attn_implementation=args.attn_implementation,
    trust_remote_code=True,
    device_map=args.device_map,
).eval()
print("hf_device_map:", getattr(model, "hf_device_map", None))

N_LAYERS = len(model.model.language_model.layers) + 1


# ============================== BBOX -> TOKEN-INDEX HELPERS (unchanged) =====
def compute_pan_and_scan_crops(width, height, min_crop_size=256, max_num_crops=4, min_ratio=1.2):
    """Replicates Gemma3ImageProcessor.pan_and_scan() to compute crop boundaries."""
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


def bbox_to_bbox(original_box):
    """Convert [[x1,y1],[x2,y2]] to (x1, y1, x2, y2)."""
    return (original_box[0][0], original_box[0][1], original_box[1][0], original_box[1][1])


def bbox_to_token_indices(x_min, y_min, x_max, y_max, original_width, original_height,
                           resized_size=896, tokens_per_side=16):
    """Convert bounding box to token indices in the 16x16 image token grid."""
    width_ratio = resized_size / original_width
    height_ratio = resized_size / original_height
    patch_size = resized_size / tokens_per_side

    x_min_token = int(x_min * width_ratio // patch_size)
    x_max_token = max(int(x_max * width_ratio // patch_size), x_min_token + 1)
    y_min_token = int(y_min * height_ratio // patch_size)
    y_max_token = max(int(y_max * height_ratio // patch_size), y_min_token + 1)

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
    offset = image_starts[image_index] + 1
    return [i + offset for i in token_indices]


def find_tokens_in_crops(input_ids, bbox, crops, first_crop_input_ids_idx,
                          vision_start_id=255999, resized_size=896, tokens_per_side=16):
    """Map a bounding box to absolute token positions across pan-and-scan crop tokens."""
    image_starts = [i for i, x in enumerate(input_ids[0].tolist()) if x == vision_start_id]
    patch_size = resized_size / tokens_per_side
    ox1, oy1, ox2, oy2 = bbox

    all_positions = []
    for crop_idx, (cx1, cy1, cx2, cy2) in enumerate(crops):
        ix1 = max(ox1, cx1); iy1 = max(oy1, cy1)
        ix2 = min(ox2, cx2); iy2 = min(oy2, cy2)
        if ix1 >= ix2 or iy1 >= iy2:
            continue

        crop_w = cx2 - cx1; crop_h = cy2 - cy1
        scale_x = resized_size / crop_w
        scale_y = resized_size / crop_h

        lx1 = (ix1 - cx1) * scale_x; ly1 = (iy1 - cy1) * scale_y
        lx2 = (ix2 - cx1) * scale_x; ly2 = (iy2 - cy1) * scale_y

        tx1 = int(lx1 // patch_size)
        ty1 = int(ly1 // patch_size)
        tx2 = max(int(lx2 // patch_size), tx1 + 1)
        ty2 = max(int(ly2 // patch_size), ty1 + 1)

        tx1 = max(0, min(tx1, tokens_per_side - 1))
        ty1 = max(0, min(ty1, tokens_per_side - 1))
        tx2 = max(0, min(tx2, tokens_per_side))
        ty2 = max(0, min(ty2, tokens_per_side))

        token_indices = [y * tokens_per_side + x for y in range(ty1, ty2) for x in range(tx1, tx2)]
        offset = image_starts[first_crop_input_ids_idx + crop_idx] + 1
        all_positions.extend(t + offset for t in token_indices)

    return all_positions


def get_target_indices(item, k, sample_input_ids, crops_tgt, first_crop_tgt_idx, tgt_global_img_idx):
    """Resolve bbox k of item's target image to absolute token positions in input_ids.
    Shared by both the logit lens and latent lens read-outs so the bbox->token
    geometry is computed exactly once per (item, k)."""
    original_box = item["tgt_positions"][k]
    bbox = bbox_to_bbox(original_box)
    tgt_image = item["tgt_image"]

    if DO_PAN_AND_SCAN and crops_tgt:
        return find_tokens_in_crops(
            input_ids=sample_input_ids, bbox=bbox, crops=crops_tgt,
            first_crop_input_ids_idx=first_crop_tgt_idx,
        )
    rel = bbox_to_token_indices(bbox[0], bbox[1], bbox[2], bbox[3], tgt_image.width, tgt_image.height)
    return get_absolute_token_positions(rel, sample_input_ids, image_index=tgt_global_img_idx)


# ============================== LOGIT LENS READ-OUT (unchanged) =============
def get_decoded_tokens(region_features):
    """Project region_features through the model's own final norm + lm_head.
    Returns a list of (top1_word, top1_prob), one per patch token in the region."""
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


# ============================== LATENT LENS (McGill-NLP) =====================
def find_bundled_corpus():
    """Locate concepts.txt bundled inside the installed latentlens package."""
    import importlib.resources as ir
    try:
        for candidate in ir.files("latentlens").iterdir():
            if candidate.name == "concepts.txt":
                return str(candidate)
    except Exception:
        pass
    pkg_dir = os.path.dirname(latentlens.__file__)
    for root, _dirs, files in os.walk(pkg_dir):
        if "concepts.txt" in files:
            return os.path.join(root, "concepts.txt")
    raise FileNotFoundError("Could not locate concepts.txt inside the latentlens package; pass args.corpus explicitly.")


def build_or_load_latentlens_index(model, tokenizer, layers):
    metadata_path = os.path.join(args.index_cache_dir, "metadata.json")
    if os.path.exists(metadata_path):
        print(f"Loading cached LatentLens index from {args.index_cache_dir}")
        return latentlens.ContextualIndex.load(args.index_cache_dir)

    corpus_path = args.corpus or find_bundled_corpus()
    print(f"Building LatentLens index from corpus: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_lines = [line.strip() for line in f if line.strip()]

    if args.max_index_sentences and len(corpus_lines) > args.max_index_sentences:
        corpus_lines = random.Random(0).sample(corpus_lines, args.max_index_sentences)
    print(f"Indexing {len(corpus_lines)} corpus sentences across {len(layers)} layers...")

    index = latentlens.build_index(
        model_name=None,
        model=model,
        tokenizer=tokenizer,
        corpus=corpus_lines,
        layers=layers,
        batch_size=args.index_batch_size,
    )
    os.makedirs(args.index_cache_dir, exist_ok=True)
    index.save(args.index_cache_dir)
    return index


latentlens_index = build_or_load_latentlens_index(model, processor.tokenizer, layers=list(range(N_LAYERS)))
torch.cuda.empty_cache()


def get_latent_tokens(region_features, layer):
    """Nearest-neighbor lookup of region_features (at this layer) against the
    LatentLens corpus index. Returns a list of (label, similarity), one per
    patch token in the region -- same shape as get_decoded_tokens.

    The corpus index contains one <bos> vector per corpus sentence (3000 of
    them), and <bos> representations cluster tightly in most middle/late
    layers -- so the literal top-1 neighbor is <bos> for the overwhelming
    majority of patches there, which carries no content information. Instead
    of always taking rank-1, walk down the top-k neighbors and take the first
    one that isn't a special/control token or pure punctuation; only fall back
    to the literal top-1 (even if that's <bos>) if every candidate in top-k is
    special/boilerplate."""
    query = region_features.to(latentlens_index.device)
    neighbor_lists = latentlens_index.search(query, top_k=args.latentlens_top_k, layers=[layer])

    token_values = []
    for patch_idx in range(region_features.shape[0]):
        neighbors = neighbor_lists[patch_idx]
        chosen = neighbors[0]
        if args.latent_skip_special:
            for n in neighbors:
                label = (n.token_str or "").strip() or (n.caption or "").strip()
                if not is_special_or_boilerplate(label):
                    chosen = n
                    break
        label = (chosen.token_str or "").strip() or (chosen.caption or "").strip()
        token_values.append((label, float(chosen.similarity)))
    return token_values


# ============================== SHARED UTILITIES (unchanged) ================
def collate_fn(batch):
    images_list = [[item["ref_image"], item["tgt_image"]] for item in batch]
    messages_list = [[
        {"role": "user", "content": [
            {"type": "image", "image": item["ref_image"]},
            {"type": "image", "image": item["tgt_image"]},
            {"type": "text", "text": item["prompt"]},
        ]}
    ] for item in batch]
    texts = processor.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True)
    inputs = processor(images_list, texts, padding=True, return_tensors="pt")
    return inputs, batch


def is_color(name):
    try:
        webcolors.name_to_hex(name)
        return True
    except ValueError:
        return False


def is_special_or_boilerplate(t_stripped):
    """True for special/control tokens (<bos>, <eos>, <pad>, <unk>, ...) and for
    pure-punctuation tokens (e.g. "'", "."). These dominate the LatentLens
    nearest-neighbor readout at many layers (e.g. <bos> matching almost every
    patch) and would otherwise make the Jaccard-distinctiveness metric look
    artificially high/low for reasons unrelated to shape distinguishability."""
    if not t_stripped:
        return True
    if re.fullmatch(r"<[^>]*>", t_stripped):
        return True
    if not any(ch.isalnum() for ch in t_stripped):
        return True
    return False


def filter_tokens(tokens):
    result = []
    for t in tokens:
        t_stripped = t.strip()
        if IGNORE_COLORS and is_color(t_stripped.lower()):
            continue
        if IGNORE_OPTIONS and t_stripped in {'A', 'B', 'C', 'D'}:
            continue
        if IGNORE_SPECIAL and is_special_or_boilerplate(t_stripped):
            continue
        result.append(t)
    return result


def jaccard_similarity(set1, set2):
    s1, s2 = set(set1), set(set2)
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 0.0


# ============================== MAIN PROBING LOOP ============================
def get_logit_and_latent_metrics(model, target_dataset, num_samples=-1, batch_size=16):
    if num_samples == -1:
        num_samples = len(target_dataset)

    subset = Subset(target_dataset, range(num_samples))
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    per_layer_jac_logit = defaultdict(list)
    per_layer_jac_latent = defaultdict(list)
    all_data_logit = []
    all_data_latent = []

    for inputs, meta_batch in tqdm(dataloader):
        inputs = inputs.to(model.device)
        B = len(meta_batch)

        # input_ids layout per sample: [ref_global, *ref_crops, tgt_global, *tgt_crops]
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

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        if any(torch.isnan(hs).any() for hs in outputs.hidden_states):
            raise RuntimeError(
                "NaN detected in hidden states. If args.dtype == 'float16', this is almost certainly "
                "Gemma's activations overflowing float16's range in later layers -- switch args.dtype "
                "to 'bfloat16' (or 'float32') and rerun."
            )

        batch_sample_logit = [[] for _ in range(B)]
        batch_sample_latent = [[] for _ in range(B)]

        for layer in range(N_LAYERS):
            layer_hs = outputs.hidden_states[layer]  # [B, seq_len, D]

            for b_idx in range(B):
                sample_hs = layer_hs[b_idx: b_idx + 1]
                sample_input_ids = inputs.input_ids[b_idx: b_idx + 1]
                crops_tgt, first_crop_tgt_idx, tgt_global_img_idx = crops_info[b_idx]

                logit_token_values, latent_token_values = [], []
                logit_tokens_only, latent_tokens_only = [], []

                for k in range(4):
                    target_indices = get_target_indices(
                        meta_batch[b_idx], k, sample_input_ids,
                        crops_tgt, first_crop_tgt_idx, tgt_global_img_idx,
                    )
                    region_features = sample_hs[0, target_indices, :]

                    tv_logit = get_decoded_tokens(region_features)
                    tv_latent = get_latent_tokens(region_features, layer)

                    logit_token_values.append(tv_logit)
                    latent_token_values.append(tv_latent)
                    logit_tokens_only.append([x[0] for x in tv_logit])
                    latent_tokens_only.append([x[0] for x in tv_latent])

                batch_sample_logit[b_idx].append(logit_token_values)
                batch_sample_latent[b_idx].append(latent_token_values)

                filtered_logit = [filter_tokens(logit_tokens_only[i]) for i in range(4)]
                filtered_latent = [filter_tokens(latent_tokens_only[i]) for i in range(4)]

                per_layer_jac_logit[layer].extend(
                    jaccard_similarity(filtered_logit[i], filtered_logit[j])
                    for i in range(4) for j in range(i + 1, 4)
                )
                per_layer_jac_latent[layer].extend(
                    jaccard_similarity(filtered_latent[i], filtered_latent[j])
                    for i in range(4) for j in range(i + 1, 4)
                )

        all_data_logit.extend(batch_sample_logit)
        all_data_latent.extend(batch_sample_latent)

    return per_layer_jac_logit, per_layer_jac_latent, all_data_logit, all_data_latent


per_layer_jac_logit, per_layer_jac_latent, all_data_logit, all_data_latent = get_logit_and_latent_metrics(
    model, test_dataset, num_samples=args.sample_size, batch_size=args.batch_size
)

layers_sorted = sorted(per_layer_jac_logit)
logit_means = [np.mean(per_layer_jac_logit[layer]) for layer in layers_sorted]
latent_means = [np.mean(per_layer_jac_latent[layer]) for layer in layers_sorted]

print("\nMean Jaccard similarity per layer (logit lens vs. latent lens):")
print(f"{'Layer':>6} | {'logit lens':>12} | {'latent lens':>12}")
for layer, lg, lt in zip(layers_sorted, logit_means, latent_means):
    print(f"{layer:>6} | {lg:>12.4f} | {lt:>12.4f}")


def plot_jaccard(layers, scores, label, color, filename):
    plt.figure()
    plt.plot(layers, scores, color=color)
    plt.xlabel("Layer")
    plt.ylabel("Mean Jaccard Similarity")
    plt.xlim(0, N_LAYERS)
    plt.ylim(0, 1)
    max_score = max(scores)
    max_score_layer = layers[scores.index(max_score)] if isinstance(scores, list) else layers[int(np.argmax(scores))]
    plt.axhline(y=max_score, alpha=0.5, color='red', linestyle='--',
                label=f'Max Jaccard: {max_score:.4f} at Layer {max_score_layer}')
    plt.title(f"{label} -- lower = more distinguishable shapes")
    plt.legend()
    plt.savefig(filename)
    plt.close()


plot_jaccard(layers_sorted, list(logit_means), "Logit Lens", "tab:blue", f"{OUTPUT_FILE_NAME}_logit_jaccard.png")
plot_jaccard(layers_sorted, list(latent_means), "Latent Lens", "tab:orange", f"{OUTPUT_FILE_NAME}_latent_jaccard.png")

# comparison plot: both lenses' Jaccard-distinctiveness curves on one axis
plt.figure(figsize=(9, 5))
plt.plot(layers_sorted, logit_means, label="Logit Lens", color="tab:blue")
plt.plot(layers_sorted, latent_means, label="Latent Lens", color="tab:orange")
plt.xlabel("Layer")
plt.ylabel("Mean Jaccard Similarity (lower = more distinguishable)")
plt.xlim(0, N_LAYERS)
plt.ylim(0, 1)
plt.title("Logit Lens vs. Latent Lens -- per-layer comparison")
plt.legend()
plt.savefig(f"{OUTPUT_FILE_NAME}_comparison.png")
plt.close()

with open(f"{OUTPUT_FILE_NAME}_logit_all_data.pkl", "wb") as f:
    pickle.dump(all_data_logit, f)
with open(f"{OUTPUT_FILE_NAME}_latent_all_data.pkl", "wb") as f:
    pickle.dump(all_data_latent, f)
with open(f"{OUTPUT_FILE_NAME}_logit_jaccard.pkl", "wb") as f:
    pickle.dump(logit_means, f)
with open(f"{OUTPUT_FILE_NAME}_latent_jaccard.pkl", "wb") as f:
    pickle.dump(latent_means, f)

print(f"\nSaved results under: {OUTPUT_FILE_NAME}_*")


# %%
###CELL 2
# ----------------------------------------------------------------------------
# Per-patch trajectory plots (reproduces logit_lens_traj.ipynb), generalized to
# work for either lens via the LENS toggle below. Paste into its own cell and
# run after Cell 1 (no GPU needed -- it only reads back the saved pickles).
# ----------------------------------------------------------------------------
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from nltk.stem import PorterStemmer

LENS = "logit"          # "logit" or "latent"
DATASET_NAME = "basic_shapes_TEST"
MODEL_PATH = "google/gemma-3-12b-it"
RESULTS_DIR = "/home/user8/VLMs-Need-Words-COLM2026/2D_shape_recognition/logit_latent_results"

data_idx = 36
option_idx = 0          # which candidate position to trace: 0=A, 1=B, 2=C, 3=D
SKIP_LAYERS = 0

file_path = f"{RESULTS_DIR}/dataset{DATASET_NAME}_model{MODEL_PATH.replace('/', '_')}_{LENS}_all_data.pkl"
with open(file_path, "rb") as f:
    target_data = pickle.load(f)

# target_data[data_idx] is a list over layers; each layer entry is a list of 4
# option entries (one per A/B/C/D bbox); each of those is a list of (label, value)
# tuples, one per image-patch token that falls inside that option's bounding box.
# For LENS="logit", value = top-1 vocabulary probability. For LENS="latent",
# value = top-1 cosine similarity to the nearest LatentLens corpus neighbor.
lens_data = [layer_data[option_idx] for layer_data in target_data[data_idx]]

VALUE_LABEL = "Confidence Score" if LENS == "logit" else "Cosine Similarity"
TITLE_SUFFIX = "Logit Lens" if LENS == "logit" else "Latent Lens"

save_dir = f"{DATASET_NAME}_{LENS}_lens_plots_item{data_idx}_option{option_idx}"
os.makedirs(save_dir, exist_ok=True)


def consecutive_run_lengths(stems):
    n = len(stems)
    run_len = [1] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and stems[j] == stems[i]:
            j += 1
        length = j - i
        for k in range(i, j):
            run_len[k] = length
        i = j
    return run_len


FONTSIZE = 24
TITLE_FONTSIZE = FONTSIZE + 6
LABEL_FONTSIZE = FONTSIZE + 0
TICK_FONTSIZE = FONTSIZE
ANNOT_FONTSIZE = FONTSIZE
LEGEND_FONTSIZE = FONTSIZE

_stemmer = PorterStemmer()


def stem(word):
    return _stemmer.stem(word.lower()) if word else ""


import warnings
warnings.filterwarnings("ignore", message=r"Glyph .* missing from font")
warnings.filterwarnings("ignore", message=r"Matplotlib currently does not support .* natively")

to_show = []  # optionally fill in patch indices you want printed for inspection

for target_token_idx in range(len(lens_data[0])):
    layers, words, values = [], [], []

    for_print = ""
    for layer in range(len(lens_data)):
        if layer < SKIP_LAYERS:
            continue
        word, value = lens_data[layer][target_token_idx]
        layers.append(layer)
        words.append(word)
        values.append(value)
        for_print += f"{word} ({value:.2f}), "

    stems = [stem(w) for w in words]

    if target_token_idx in to_show:
        print(f"Patch {target_token_idx} words: {for_print}")
        print()

    run_lengths = consecutive_run_lengths(stems)

    annotate_indices = set()
    i = 0
    while i < len(stems):
        j = i + 1
        while j < len(stems) and stems[j] == stems[i]:
            j += 1
        span_len = j - i
        if span_len == 1:
            annotate_indices.add(i)
        else:
            for idx in range(i, j, 4):
                annotate_indices.add(idx)
        i = j

    GREY = "#CCCCCC"
    BG_COLORS = ["#F08080", "#87CEEB", "#F5DEB3", "#90EE90", "#DDA0DD", "#FFB347"]
    POINT_COLORS = ["#E05555", "#5B8DB8", "#6DAE81", "#9B72AA", "#D4A843", "#4DBECC"]

    qualified_stems = list(dict.fromkeys(s for s, r in zip(stems, run_lengths) if r >= 2))
    qualified_words = list(dict.fromkeys(w for w, r in zip(words, run_lengths) if r >= 2))
    stem_color = {s: BG_COLORS[i % len(BG_COLORS)] for i, s in enumerate(qualified_stems)}
    word_color = {w: POINT_COLORS[i % len(POINT_COLORS)] for i, w in enumerate(qualified_words)}

    def get_stem_color(i): return stem_color.get(stems[i], GREY)
    def get_word_color(i): return word_color.get(words[i], GREY)

    fig, ax = plt.subplots(figsize=(16, 7))

    i = 0
    while i < len(stems):
        j = i + 1
        while j < len(stems) and stems[j] == stems[i]:
            j += 1
        x0 = layers[i] - (0.5 if i == 0 else (layers[i] - layers[i - 1]) / 2)
        x1 = layers[j - 1] + (0.5 if j == len(layers) else (layers[j] - layers[j - 1]) / 2)
        color = get_stem_color(i)
        ax.axvspan(x0, x1, alpha=0.25, color=color, linewidth=0, zorder=0)
        i = j

    ax.plot(layers, values, color="#888888", linewidth=1.8, zorder=2)

    v_loc = 60
    for idx, (layer, word, value) in enumerate(zip(layers, words, values)):
        ax.plot(layer, value, "o", markersize=16, color=get_word_color(idx),
                markeredgecolor="white", markeredgewidth=1.8, zorder=3)
        if idx in annotate_indices:
            ax.annotate(
                f"{word}", xy=(layer, value), xytext=(0, v_loc), textcoords="offset points",
                ha="center", va="bottom" if idx % 2 == 0 else "top",
                fontsize=ANNOT_FONTSIZE,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=get_stem_color(idx),
                          edgecolor="#999999", alpha=0.9, linewidth=0.8),
                arrowprops=dict(arrowstyle="-|>", color="#777777", lw=0.9),
                zorder=4,
            )
            v_loc = -v_loc

    ax.legend(
        handles=[mpatches.Patch(facecolor=stem_color[s], alpha=0.6, label=f"{s}") for s in qualified_stems],
        fontsize=LEGEND_FONTSIZE,
    )

    ax.set(
        xlabel="Layer",
        ylabel=VALUE_LABEL,
        title=f"Patch {target_token_idx} -- {TITLE_SUFFIX} Across Layers",
        xlim=(layers[0] - 0.8, layers[-1] + 0.8),
        ylim=(-0.08, 1.25),
    )

    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.title.set_size(TITLE_FONTSIZE)
    ax.xaxis.label.set_size(LABEL_FONTSIZE)
    ax.yaxis.label.set_size(LABEL_FONTSIZE)

    if len(layers) > 0:
        xticks = [layers[i] for i in range(0, len(layers), 5)]
        if layers[-1] not in xticks:
            xticks.append(layers[-1])
        ax.set_xticks(xticks)

    ax.grid(alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.ylim(-0.5, 1.50)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/patch{target_token_idx}.pdf", dpi=100, bbox_inches="tight")
    plt.close()

print(f"Saved {LENS}-lens trajectory plots to {save_dir}/")
