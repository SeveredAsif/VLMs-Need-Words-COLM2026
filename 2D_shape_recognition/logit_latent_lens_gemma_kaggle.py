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
    model_path="google/gemma-3-12b-it",     # passed to from_pretrained -- on Kaggle with a "Model" input this
                                             # will instead be a long local mount path; that's fine for loading,
                                             # but see model_tag below for why it must NOT be used in filenames
    model_tag="gemma-3-12b-it",              # short, stable name used ONLY for output filenames (see OUTPUT_FILE_NAME
                                             # below and Cell 2's MODEL_TAG) -- keep this the same across Cell 1/Cell 2
                                             # runs regardless of what args.model_path happens to be on this platform
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
    add_shape_corpus=True,                  # force-include generated shape/color sentences (see generate_shape_corpus) --
                                             # concepts.txt alone has ~0 real coverage of words like "hexagon"/"pentagon"
    max_index_sentences=3000,               # subsample concepts.txt for a fast index build
    index_cache_dir="/home/user8/VLMs-Need-Words-COLM2026/latentlens_index_v2",  # _v2: old cache was built
                                             # without the shape/color corpus and would be silently reused otherwise
    latentlens_top_k=20,  # bumped up from 5 -- gives get_latent_tokens more candidates to skip
                           # past <bos>/punctuation before falling back to it (see IGNORE_SPECIAL)
    index_batch_size=32,
    output_root="/home/user8/VLMs-Need-Words-COLM2026/2D_shape_recognition/logit_latent_results",
)

IGNORE_COLORS = args.ignore_colors
IGNORE_OPTIONS = args.ignore_options
IGNORE_SPECIAL = args.ignore_special
DO_PAN_AND_SCAN = args.ps

OUTPUT_FILE_NAME = f"{args.output_root}/dataset{args.dataset_name}_model{args.model_tag}"
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


SHAPE_CORPUS_COLORS = ["black", "white", "red", "blue", "green", "yellow",
                        "orange", "purple", "pink", "brown", "grey", "cyan"]
SHAPE_CORPUS_TEMPLATES = [
    "This is a picture of a {color} {shape}.",
    "The image shows a {color} {shape} shape.",
    "A {color} {shape} is drawn on the page.",
    "Look at the {color} {shape} in the corner.",
    "There is a {shape} colored {color} here.",
    "The {shape} in the picture appears in {color}.",
    "Someone drew a small {color} {shape} on the sheet.",
    "The shape is a {shape}, and its color is {color}.",
]


def generate_shape_corpus(base_shapes):
    """Guaranteed-coverage sentences for this dataset's own shape vocabulary
    (e.g. "hexagon", "pentagon"), since the bundled general-knowledge corpus
    has zero or near-zero real coverage of geometric shape words -- see the
    coverage check this feeds into. Multiple templates/colors per shape so the
    nearest-neighbor search still has some contextual variety to choose from,
    not just one canonical sentence per word."""
    lines = []
    for shape in base_shapes:
        for color in SHAPE_CORPUS_COLORS:
            for template in SHAPE_CORPUS_TEMPLATES:
                lines.append(template.format(color=color, shape=shape))
    return lines


def load_indexed_corpus_lines():
    """Reproduces exactly the same corpus mix that build_or_load_latentlens_index
    feeds into the index -- so coverage can be checked even when the index was
    loaded from cache, and without needing the model/GPU at all.

    The final corpus = ALL of the generated shape/color sentences (forced in,
    never subsampled away) + a seed=0 random sample of the bundled general
    corpus filling out the remaining budget up to args.max_index_sentences.
    Forcing the shape sentences in is necessary: with plain random sampling
    down to 3000 lines out of 117k, a rare word like "hexagon" would almost
    certainly get sampled out entirely even if a handful of instances existed."""
    corpus_path = args.corpus or find_bundled_corpus()
    with open(corpus_path, "r", encoding="utf-8") as f:
        general_lines = [line.strip() for line in f if line.strip()]
    full_size = len(general_lines)

    base_shapes = sorted({st.split("_")[0] for item in test_dataset.data
                           for st in item.get("shape_types", [])})
    shape_lines = generate_shape_corpus(base_shapes) if args.add_shape_corpus else []

    if args.max_index_sentences:
        budget = max(0, args.max_index_sentences - len(shape_lines))
        general_sample = (random.Random(0).sample(general_lines, budget)
                           if budget < len(general_lines) else list(general_lines))
    else:
        general_sample = general_lines

    corpus_lines = general_sample + shape_lines
    return corpus_lines, full_size, corpus_path


def check_corpus_coverage(corpus_lines, watchlist):
    """Prints how many (and which) of the indexed sentences contain each
    watchlist word, as a whole word (not substring -- "star" matching inside
    "starting" would be a false positive). If a word has zero hits, LatentLens
    can NEVER return it as a neighbor -- not "unlikely", literally excluded
    from the search space, since the index only ever contains tokens that
    appear somewhere in this corpus."""
    print(f"\nCorpus coverage check ({len(corpus_lines)} indexed sentences):")
    missing = []
    for word in watchlist:
        pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
        hits = [line for line in corpus_lines if pattern.search(line)]
        print(f"  {word!r:12s} -> {len(hits)} sentence(s)" + (f"   e.g. {hits[0][:80]!r}" if hits else ""))
        if not hits:
            missing.append(word)
    if missing:
        print(f"  WARNING: these words CANNOT appear as latent-lens neighbors at all: {missing}")
    return missing


def build_or_load_latentlens_index(model, tokenizer, layers):
    metadata_path = os.path.join(args.index_cache_dir, "metadata.json")
    if os.path.exists(metadata_path):
        print(f"Loading cached LatentLens index from {args.index_cache_dir}")
        return latentlens.ContextualIndex.from_directory(args.index_cache_dir)

    corpus_lines, full_size, corpus_path = load_indexed_corpus_lines()
    print(f"Building LatentLens index from corpus: {corpus_path} ({full_size} lines total)")
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

# Sanity check: can the indexed corpus even produce this dataset's shape/color
# words as neighbors at all? Built from the dataset's own shape_types (e.g.
# "diamond_black" -> "diamond", "black"), so this adapts to whichever dataset
# folder args.dataset_name points at.
_watchlist = sorted({part for item in test_dataset.data
                      for st in item.get("shape_types", [])
                      for part in st.split("_") if part})
check_corpus_coverage(load_indexed_corpus_lines()[0], _watchlist)


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
###CELL 1B -- Cross-layer LatentLens matching (image layer N vs. text layer M)
# ----------------------------------------------------------------------------
# Cell 1 above only ever searches image layer N's region_features against the
# corpus index's layer-N embeddings (the diagonal). This cell builds the full
# N x M grid instead: does image-layer-N's representation match EARLY text
# layers best (image processed "like text" early -- interesting), or only ever
# match LATE text layers regardless of N (CLIP-style alignment -- not
# interesting, already expected)? Run this in the same session right after
# Cell 1 -- it reuses `model`, `latentlens_index`, `test_dataset`, `N_LAYERS`,
# `OUTPUT_FILE_NAME`, and the shared helper functions defined above.
#
# Step-0 finding (see task doc): *_latent_all_data.pkl only ever stored top-1
# (label, similarity) per patch, not raw region_features -- so this requires
# one more GPU forward pass. The corpus/text side needs NO rebuild (it's
# loaded from latentlens_index_v2/ on disk); only the image side needs it,
# and only ONE pass total (output_hidden_states=True already returns every
# layer), not one pass per (N, M) cell.
# ----------------------------------------------------------------------------

IMAGE_LAYERS_TO_TEST = sorted(set(list(range(0, N_LAYERS, 4)) + [N_LAYERS - 1]))
TEXT_LAYERS_TO_TEST = sorted(set(list(range(0, N_LAYERS, 4)) + [N_LAYERS - 1]))
CROSSLAYER_NUM_SAMPLES = 20     # coarse grid over fewer samples than the full Cell 1 run -- raise once tuned
CROSSLAYER_SAVE_REGION_FEATURES = True  # cache raw region_features to disk -- future re-analysis needs no GPU/model

print(f"\nCross-layer grid: {len(IMAGE_LAYERS_TO_TEST)} image layers x {len(TEXT_LAYERS_TO_TEST)} text layers "
      f"= {len(IMAGE_LAYERS_TO_TEST) * len(TEXT_LAYERS_TO_TEST)} cells, over {CROSSLAYER_NUM_SAMPLES} samples.")


def get_target_label_from_shape_type(shape_type):
    """'diamond_black' -> 'diamond' -- the ground-truth shape word for an option's bbox."""
    return shape_type.split("_")[0].strip().lower()


def cross_layer_latent_lookup(region_features, text_layer):
    """Same skip-to-next-non-special logic as get_latent_tokens, but the corpus
    layer searched (text_layer) is passed explicitly instead of always being
    the same layer the region_features came from -- that's the whole point of
    this cell. Returns (label, similarity, full_neighbor_list) per patch, so
    find_target_rank can also inspect the rest of the top-k candidates."""
    query = region_features.to(latentlens_index.device)
    neighbor_lists = latentlens_index.search(query, top_k=args.latentlens_top_k, layers=[text_layer])

    results = []
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
        results.append((label, float(chosen.similarity), neighbors))
    return results


def find_target_rank(neighbors, target_word):
    """1-indexed rank + similarity of target_word among this patch's top-k
    neighbors, or (None, None) if it never appears there at all."""
    for rank, n in enumerate(neighbors, start=1):
        label = (n.token_str or "").strip().lower()
        if label == target_word:
            return rank, float(n.similarity)
    return None, None


jaccard_grid = defaultdict(list)                  # (N, M) -> jaccard scores across samples/option-pairs
target_conf_grid = defaultdict(list)              # (N, M) -> similarities, only where target word was found in top-k
target_found_grid = defaultdict(lambda: [0, 0])   # (N, M) -> [found_count, total_count]

region_features_cache = {}   # (sample_idx, image_layer, option_idx) -> float16 np.ndarray [num_patches, hidden_dim]
estimated_bytes = 0

crosslayer_subset = Subset(test_dataset, range(min(CROSSLAYER_NUM_SAMPLES, len(test_dataset))))
crosslayer_loader = DataLoader(crosslayer_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

sample_offset = 0
for inputs, meta_batch in tqdm(crosslayer_loader, desc="Cross-layer grid"):
    inputs = inputs.to(model.device)
    B = len(meta_batch)

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

    if any(torch.isnan(outputs.hidden_states[n]).any() for n in IMAGE_LAYERS_TO_TEST):
        raise RuntimeError("NaN detected in hidden states during cross-layer pass.")

    for b_idx in range(B):
        sample_idx = sample_offset + b_idx
        sample_input_ids = inputs.input_ids[b_idx: b_idx + 1]
        crops_tgt, first_crop_tgt_idx, tgt_global_img_idx = crops_info[b_idx]
        item = meta_batch[b_idx]

        # bbox -> token-position resolution doesn't depend on layer, only on k -- compute once per option
        target_indices_per_option = [
            get_target_indices(item, k, sample_input_ids, crops_tgt, first_crop_tgt_idx, tgt_global_img_idx)
            for k in range(4)
        ]
        target_words = [get_target_label_from_shape_type(st) for st in item["shape_types"]]

        for N in IMAGE_LAYERS_TO_TEST:
            layer_hs = outputs.hidden_states[N][b_idx: b_idx + 1]

            region_features_per_option = []
            for k in range(4):
                region_features = layer_hs[0, target_indices_per_option[k], :]
                region_features_per_option.append(region_features)
                if CROSSLAYER_SAVE_REGION_FEATURES:
                    arr = region_features.detach().to(torch.float16).cpu().numpy()
                    region_features_cache[(sample_idx, N, k)] = arr
                    estimated_bytes += arr.nbytes

            for M in TEXT_LAYERS_TO_TEST:
                labels_per_option = []
                for k in range(4):
                    results = cross_layer_latent_lookup(region_features_per_option[k], M)
                    labels_per_option.append([r[0] for r in results])

                    target_word = target_words[k]
                    for _, _, neighbors in results:
                        rank, sim = find_target_rank(neighbors, target_word)
                        target_found_grid[(N, M)][1] += 1
                        if rank is not None:
                            target_found_grid[(N, M)][0] += 1
                            target_conf_grid[(N, M)].append(sim)

                filtered = [filter_tokens(labels_per_option[i]) for i in range(4)]
                jaccard_grid[(N, M)].extend(
                    jaccard_similarity(filtered[i], filtered[j])
                    for i in range(4) for j in range(i + 1, 4)
                )

    sample_offset += B

print(f"\nCached region_features for {len(region_features_cache)} (sample, image_layer, option) entries, "
      f"~{estimated_bytes / 1e9:.2f} GB (float16).")
if estimated_bytes > 3e9:
    print("WARNING: region_features cache is > 3GB -- consider reducing CROSSLAYER_NUM_SAMPLES, "
          "the layer-grid stride, or setting CROSSLAYER_SAVE_REGION_FEATURES=False.")

if CROSSLAYER_SAVE_REGION_FEATURES:
    region_features_path = f"{OUTPUT_FILE_NAME}_region_features.pt"
    torch.save(region_features_cache, region_features_path)
    print(f"Saved raw region_features to {region_features_path}")

# ------------------------------ aggregate the grid ---------------------------
jaccard_mean_grid = np.full((len(IMAGE_LAYERS_TO_TEST), len(TEXT_LAYERS_TO_TEST)), np.nan)
target_conf_mean_grid = np.full((len(IMAGE_LAYERS_TO_TEST), len(TEXT_LAYERS_TO_TEST)), np.nan)
target_found_rate_grid = np.full((len(IMAGE_LAYERS_TO_TEST), len(TEXT_LAYERS_TO_TEST)), np.nan)

for i, N in enumerate(IMAGE_LAYERS_TO_TEST):
    for j, M in enumerate(TEXT_LAYERS_TO_TEST):
        scores = jaccard_grid.get((N, M), [])
        if scores:
            jaccard_mean_grid[i, j] = np.mean(scores)
        confs = target_conf_grid.get((N, M), [])
        if confs:
            target_conf_mean_grid[i, j] = np.mean(confs)
        found, total = target_found_grid.get((N, M), [0, 0])
        if total:
            target_found_rate_grid[i, j] = found / total

crosslayer_grid_data = {
    "image_layers": IMAGE_LAYERS_TO_TEST,
    "text_layers": TEXT_LAYERS_TO_TEST,
    "jaccard_mean_grid": jaccard_mean_grid,
    "target_conf_mean_grid": target_conf_mean_grid,
    "target_found_rate_grid": target_found_rate_grid,
    "num_samples": CROSSLAYER_NUM_SAMPLES,
}
with open(f"{OUTPUT_FILE_NAME}_crosslayer_grid.pkl", "wb") as f:
    pickle.dump(crosslayer_grid_data, f)
print(f"Saved cross-layer grid data to {OUTPUT_FILE_NAME}_crosslayer_grid.pkl "
      f"(regenerate/restyle heatmaps from this later without recomputation)")


# ------------------------------ heatmaps + line plot --------------------------
def plot_crosslayer_heatmap(grid, image_layers, text_layers, title, cbar_label, filename, cmap="viridis"):
    plt.figure(figsize=(8, 7))
    im = plt.imshow(grid, origin="lower", aspect="auto", cmap=cmap,
                     extent=[text_layers[0] - 0.5, text_layers[-1] + 0.5,
                             image_layers[0] - 0.5, image_layers[-1] + 0.5])
    plt.colorbar(im, label=cbar_label)
    plt.xlabel("Text layer M (LatentLens corpus)")
    plt.ylabel("Image layer N (Gemma hidden state)")
    plt.title(title)
    lo = min(image_layers[0], text_layers[0])
    hi = max(image_layers[-1], text_layers[-1])
    plt.plot([lo, hi], [lo, hi], color="red", linestyle="--", alpha=0.6, label="N = M (diagonal)")
    plt.legend(loc="upper left", fontsize=8)
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close()


plot_crosslayer_heatmap(
    jaccard_mean_grid, IMAGE_LAYERS_TO_TEST, TEXT_LAYERS_TO_TEST,
    "Cross-layer Jaccard distinctiveness (lower = more distinguishable shapes)",
    "Mean Jaccard similarity", f"{OUTPUT_FILE_NAME}_crosslayer_jaccard_heatmap.png", cmap="viridis_r",
)
plot_crosslayer_heatmap(
    target_conf_mean_grid, IMAGE_LAYERS_TO_TEST, TEXT_LAYERS_TO_TEST,
    "Cross-layer target-label confidence (mean similarity, when target word found in top-k)",
    "Mean similarity", f"{OUTPUT_FILE_NAME}_crosslayer_targetconf_heatmap.png", cmap="viridis",
)

best_M_per_N = []
for i, N in enumerate(IMAGE_LAYERS_TO_TEST):
    row = target_conf_mean_grid[i, :]
    best_M_per_N.append(np.nan if np.all(np.isnan(row)) else TEXT_LAYERS_TO_TEST[int(np.nanargmax(row))])

plt.figure(figsize=(7, 6))
plt.plot(IMAGE_LAYERS_TO_TEST, best_M_per_N, marker="o", color="tab:purple", label="M* (best-matching text layer)")
plt.plot([IMAGE_LAYERS_TO_TEST[0], IMAGE_LAYERS_TO_TEST[-1]],
         [IMAGE_LAYERS_TO_TEST[0], IMAGE_LAYERS_TO_TEST[-1]],
         color="red", linestyle="--", alpha=0.6, label="M* = N (diagonal)")
plt.xlabel("Image layer N")
plt.ylabel("Best-matching text layer M*")
plt.title("Best text layer per image layer -- diagonal supports 'processed like text early'")
plt.legend()
plt.savefig(f"{OUTPUT_FILE_NAME}_crosslayer_bestM_line.png", dpi=120, bbox_inches="tight")
plt.close()

print("Saved cross-layer heatmaps (_crosslayer_jaccard_heatmap.png, _crosslayer_targetconf_heatmap.png) "
      "and the M* vs. N line plot (_crosslayer_bestM_line.png).")


# %%
###CELL 2
# ----------------------------------------------------------------------------
# Per-patch trajectory plots (reproduces logit_lens_traj.ipynb). Runs for BOTH
# lenses in one execution (LENSES_TO_RUN below) so you don't have to manually
# flip LENS and rerun the cell a second time. Paste into its own cell and run
# after Cell 1 (no GPU needed -- it only reads back the saved pickles).
# ----------------------------------------------------------------------------
import os
import json
import math
import pickle
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from nltk.stem import PorterStemmer

LENSES_TO_RUN = ["logit", "latent"]   # runs both, back to back, in this one cell execution
DATASET_NAME = "basic_shapes_TEST"
MODEL_TAG = "gemma-3-12b-it"    # must match args.model_tag from Cell 1 exactly -- NOT args.model_path
                                # (on Kaggle, model_path is a long local mount path, not this clean tag)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logit_latent_results")

# Local clone of the dataset -- only used to look up which image-grid cell each
# "Patch N" actually is (answers.json + the target image's pixel size). No GPU
# or model needed for this -- it's pure bbox geometry, recomputed locally.
DATASET_ROOT_LOCAL = os.path.dirname(os.path.abspath(__file__))
DO_PAN_AND_SCAN = True   # must match args.ps used when Cell 1 was run

data_idx = 36
option_idx = 0          # which candidate position to trace: 0=A, 1=B, 2=C, 3=D
OPTION_LETTERS = ["A", "B", "C", "D"]
SKIP_LAYERS = 0

# Logit-lens values are already softmax probabilities over the vocab. LatentLens
# values are raw cosine similarities (roughly [-1, 1], usually compressed into a
# narrow band like 0.75-0.95 in practice) -- not on a comparable scale, so a
# similarity of 0.3 does NOT mean the same thing as a logit-lens probability of
# 0.3. To compare fairly, softmax-normalize the similarities ACROSS THE LAYER
# AXIS, independently per patch/option (NOT across patches, and NOT across the
# top-k neighbor candidates -- only the top-1 value survived into the saved
# pickle, so a "softmax over candidates" would need Task A's raw-feature
# caching to also store the full top-k list per layer; that's a possible
# future addition, not implemented here).
SOFTMAX_TEMPERATURE = 0.1   # tune this: raw similarities are narrowly distributed, so T=1.0 produces an
                            # almost-flat softmax that hides structure -- start around 0.05-0.2


def softmax_1d(x, temperature=1.0):
    x = np.array(x, dtype=np.float64) / temperature
    x = x - x.max()  # numerical stability
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


# ============================== "Patch N" -> image-grid position =============
# Reproduces the exact same bbox -> 16x16-grid-cell geometry used at inference
# time (compute_pan_and_scan_crops / bbox_to_token_indices / find_tokens_in_crops
# in logit_latent_lens_gemma_kaggle.py), just to recover *where in the image*
# each "Patch N" index actually is. Patch indices are in raster (row-major)
# order, per crop, in the same crop order the model saw them in.
def compute_pan_and_scan_crops(width, height, min_crop_size=256, max_num_crops=4, min_ratio=1.2):
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


def bbox_grid_cells(bbox, width, height, resized_size=896, tokens_per_side=16):
    """(row, col) grid cells for a bbox in the *global* (non-cropped) 16x16 grid."""
    x_min, y_min, x_max, y_max = bbox
    width_ratio = resized_size / width
    height_ratio = resized_size / height
    patch_size = resized_size / tokens_per_side

    x_min_t = int(x_min * width_ratio // patch_size)
    x_max_t = max(int(x_max * width_ratio // patch_size), x_min_t + 1)
    y_min_t = int(y_min * height_ratio // patch_size)
    y_max_t = max(int(y_max * height_ratio // patch_size), y_min_t + 1)

    x_min_t = max(0, min(x_min_t, tokens_per_side - 1))
    y_min_t = max(0, min(y_min_t, tokens_per_side - 1))
    x_max_t = max(0, min(x_max_t, tokens_per_side))
    y_max_t = max(0, min(y_max_t, tokens_per_side))

    return [{"crop": None, "row": y, "col": x}
            for y in range(y_min_t, y_max_t) for x in range(x_min_t, x_max_t)]


def bbox_grid_cells_in_crops(bbox, crops, resized_size=896, tokens_per_side=16):
    """Same as bbox_grid_cells, but per pan-and-scan crop, in crop order --
    matches find_tokens_in_crops' iteration order exactly."""
    ox1, oy1, ox2, oy2 = bbox
    patch_size = resized_size / tokens_per_side
    cells = []
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
        tx1 = int(lx1 // patch_size); ty1 = int(ly1 // patch_size)
        tx2 = max(int(lx2 // patch_size), tx1 + 1)
        ty2 = max(int(ly2 // patch_size), ty1 + 1)
        tx1 = max(0, min(tx1, tokens_per_side - 1))
        ty1 = max(0, min(ty1, tokens_per_side - 1))
        tx2 = max(0, min(tx2, tokens_per_side))
        ty2 = max(0, min(ty2, tokens_per_side))
        for y in range(ty1, ty2):
            for x in range(tx1, tx2):
                cells.append({"crop": crop_idx + 1, "n_crops": len(crops), "row": y, "col": x})
    return cells


def describe_patches(dataset_name, dataset_root, data_idx, option_idx, do_pan_and_scan):
    """Returns a list of human-readable strings, one per patch index, describing
    where that patch sits in the target image -- or None if it can't be resolved
    (e.g. dataset not available locally)."""
    try:
        answers_path = os.path.join(dataset_root, dataset_name, "answers.json")
        with open(answers_path, "r") as f:
            answers = json.load(f)
        item = answers[data_idx]
        tgt_image_path = os.path.join(dataset_root, item["tgt_image_path"])
        with Image.open(tgt_image_path) as im:
            width, height = im.size
        box = item["tgt_positions" if "tgt_positions" in item else "tgt_pixel_positions"][option_idx]
        bbox = (box[0][0], box[0][1], box[1][0], box[1][1])

        crops = compute_pan_and_scan_crops(width, height) if do_pan_and_scan else []
        cells = bbox_grid_cells_in_crops(bbox, crops) if crops else bbox_grid_cells(bbox, width, height)

        descs = []
        for c in cells:
            if c["crop"] is None:
                descs.append(f"row {c['row']}, col {c['col']} (full image)")
            else:
                descs.append(f"row {c['row']}, col {c['col']} (crop {c['crop']}/{c['n_crops']})")
        return descs
    except Exception as e:
        print(f"[patch-position lookup skipped: {e}]")
        return None


# Lens-independent (same bbox/geometry for both logit and latent), so computed
# once here rather than inside the per-lens loop below.
patch_descriptions = describe_patches(DATASET_NAME, DATASET_ROOT_LOCAL, data_idx, option_idx, DO_PAN_AND_SCAN)


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


ANNOT_WRAP_WIDTH = 26       # chars per line inside an annotation bubble
ANNOT_MAX_CHARS = 110       # truncate very long corpus sentences beyond this

_stemmer = PorterStemmer()


def stem(word):
    return _stemmer.stem(word.lower()) if word else ""


def annotation_text(word, is_sentence_lens):
    """Display text for an annotation bubble: word/token as-is for logit lens;
    wrapped + truncated for latent lens, since a label there can be a full
    corpus sentence instead of a single word."""
    if not is_sentence_lens:
        return word
    text = word if len(word) <= ANNOT_MAX_CHARS else word[:ANNOT_MAX_CHARS - 3] + "..."
    return textwrap.fill(text, width=ANNOT_WRAP_WIDTH)


import warnings
warnings.filterwarnings("ignore", message=r"Glyph .* missing from font")
warnings.filterwarnings("ignore", message=r"Matplotlib currently does not support .* natively")

to_show = []  # optionally fill in patch indices you want printed for inspection

for LENS in LENSES_TO_RUN:
    print(f"\n=== Generating {LENS}-lens trajectory plots ===")

    # Latent-lens labels can be whole corpus sentences (not single words), so
    # they need a smaller font, a wider figure, and wrapped/truncated
    # annotation text to stay readable -- logit-lens labels are always single
    # vocabulary tokens.
    IS_SENTENCE_LENS = (LENS == "latent")
    FONTSIZE = 24 if not IS_SENTENCE_LENS else 13
    TITLE_FONTSIZE = FONTSIZE + 6
    LABEL_FONTSIZE = FONTSIZE + (0 if not IS_SENTENCE_LENS else 6)
    TICK_FONTSIZE = FONTSIZE + (0 if not IS_SENTENCE_LENS else 6)
    ANNOT_FONTSIZE = FONTSIZE
    LEGEND_FONTSIZE = FONTSIZE

    file_path = f"{RESULTS_DIR}/dataset{DATASET_NAME}_model{MODEL_TAG}_{LENS}_all_data.pkl"
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

    lens_patch_descriptions = patch_descriptions
    if lens_patch_descriptions is not None and len(lens_patch_descriptions) != len(lens_data[0]):
        print(f"[warning: recomputed patch count doesn't match saved {LENS} data -- "
              "DO_PAN_AND_SCAN above may not match how Cell 1 was actually run]")
        lens_patch_descriptions = None

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

        # Grouping key for the color bands / legend: stemming works well for single
        # vocabulary words (logit lens). For latent-lens sentences, stemming a whole
        # sentence buys nothing -- group by the exact label instead, so a band only
        # forms where the lens returned the literal same neighbor across layers.
        stems = [stem(w) for w in words] if not IS_SENTENCE_LENS else list(words)

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

        softmax_values = softmax_1d(values, temperature=SOFTMAX_TEMPERATURE)
        print(f"Patch {target_token_idx} ({LENS} lens): softmax range = "
              f"[{softmax_values.min():.4f}, {softmax_values.max():.4f}] (T={SOFTMAX_TEMPERATURE})")

        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(16, 9) if not IS_SENTENCE_LENS else (22, 12),
            sharex=True, gridspec_kw={"height_ratios": [3, 1.3]},
        )

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

        v_loc = 60 if not IS_SENTENCE_LENS else 90
        for idx, (layer, word, value) in enumerate(zip(layers, words, values)):
            ax.plot(layer, value, "o", markersize=16, color=get_word_color(idx),
                    markeredgecolor="white", markeredgewidth=1.8, zorder=3)
            if idx in annotate_indices:
                ax.annotate(
                    annotation_text(word, IS_SENTENCE_LENS), xy=(layer, value), xytext=(0, v_loc),
                    textcoords="offset points",
                    ha="center", va="bottom" if idx % 2 == 0 else "top",
                    fontsize=ANNOT_FONTSIZE,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor=get_stem_color(idx),
                              edgecolor="#999999", alpha=0.9, linewidth=0.8),
                    arrowprops=dict(arrowstyle="-|>", color="#777777", lw=0.9),
                    zorder=4,
                )
                v_loc = -v_loc

        def legend_label(s):
            return s if len(s) <= 55 else s[:52] + "..."

        ax.legend(
            handles=[mpatches.Patch(facecolor=stem_color[s], alpha=0.6, label=legend_label(s)) for s in qualified_stems],
            fontsize=LEGEND_FONTSIZE,
            loc="upper left" if not IS_SENTENCE_LENS else "lower left",
        )

        # "Patch N" = the Nth image-patch token (in raster/top-left-to-bottom-right
        # order) that falls inside option {LETTER}'s bounding box in the TARGET
        # image -- NOT a position in the reference image, and not a token index in
        # the model's full input sequence. A bbox usually spans several such grid
        # cells (see subtitle below), so there is one trajectory plot per cell.
        option_letter = OPTION_LETTERS[option_idx] if option_idx < len(OPTION_LETTERS) else str(option_idx)
        if lens_patch_descriptions is not None:
            where = lens_patch_descriptions[target_token_idx]
            subtitle = f"image-grid cell: {where} of option {option_letter}'s box in the target image"
        else:
            subtitle = (f"{target_token_idx}-th image-grid cell (raster order) inside option "
                        f"{option_letter}'s box in the target image")

        ax.set(
            ylabel=VALUE_LABEL,  # xlabel omitted here -- shared x-axis, "Layer" label lives on ax2 below
            xlim=(layers[0] - 0.8, layers[-1] + 0.8),
            ylim=(-0.08, 1.25),
        )
        ax.set_title(f"Patch {target_token_idx} -- {TITLE_SUFFIX} Across Layers\n({subtitle})",
                     fontsize=TITLE_FONTSIZE)

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
        ax.set_ylim(-0.5, 1.50)

        # Second panel: same trajectory, softmax-normalized across the layer axis so
        # it's on a true [0, 1] probability scale -- directly comparable to logit
        # lens's already-probabilistic values, unlike the raw panel above (cosine
        # similarities for latent lens are not probabilities and aren't comparable
        # 1:1 with logit-lens probabilities on the same numeric scale).
        ax2.plot(layers, softmax_values, color="#444444", linewidth=1.8, marker="o",
                 markersize=6, zorder=2)
        for idx, (layer, sm_val) in enumerate(zip(layers, softmax_values)):
            ax2.plot(layer, sm_val, "o", markersize=8, color=get_word_color(idx),
                      markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Layer", fontsize=LABEL_FONTSIZE)
        ax2.set_ylabel(f"Softmax Probability\n(T={SOFTMAX_TEMPERATURE})", fontsize=max(LABEL_FONTSIZE - 6, 9))
        ax2.tick_params(axis="both", labelsize=max(TICK_FONTSIZE - 6, 9))
        ax2.grid(alpha=0.25, linestyle="--")
        ax2.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/patch{target_token_idx}.pdf", dpi=100, bbox_inches="tight")
        plt.close()

    print(f"Saved {LENS}-lens trajectory plots to {save_dir}/")

print("\nDone -- both logit and latent trajectory plots generated in this one run.")
