"""
Probe where the "target_word" concept (e.g. " sphere") emerges inside a VLM's
layers, using the CLEVR-Ladder style datasets (L1_train.json / L*_val.json).

Two lenses are computed per layer, over the image-patch tokens that cover the
scene object matching `target_word`:

  - Logit lens:  hidden_state -> final norm -> lm_head -> softmax
                 -> probability mass on the target_word token
                 (+ fraction of region patches whose top-1 decoded token IS
                 the target word).
  - Latent lens: real McGill-NLP LatentLens (https://github.com/McGill-NLP/latentlens).
                 Region hidden states are searched, per layer, against a
                 ContextualIndex built from a text corpus run through this
                 same model -- i.e. nearest-neighbor retrieval in embedding
                 space rather than a vocab projection. We record, per layer,
                 the fraction of region patches whose retrieved neighbors
                 mention the target word, and the mean top-1 similarity.
                 No prebuilt index ships for Gemma-3, so the index is built
                 once from a text corpus and cached to disk for reuse across
                 runs (see --index_cache_dir). If --corpus isn't given, the
                 `concepts.txt` file bundled inside the installed `latentlens`
                 package is used automatically -- no need to clone that repo.

Only "yes" examples are used, since those are the ones where the queried
shape actually exists in the scene (so a bbox is available to localize it).

This file is standalone -- no other repo files are imported, so the whole
thing can be pasted into a single Kaggle/Jupyter notebook cell and run with
Shift+Enter, no .py file or CLI needed. Configuration is the plain `args =
SimpleNamespace(...)` block below (edit the values there directly) rather
than argparse, since argparse reads sys.argv, which inside a notebook kernel
is the kernel launcher's own arguments, not yours.

First cell:
    !pip install -q -U transformers accelerate torch pillow tqdm matplotlib latentlens

Second cell: paste this whole file, edit the `args = SimpleNamespace(...)` block
near the top (dataset_json / model_path / device_map / etc.), then run the cell.
For Kaggle GPU T4 x2 with a 12b+ model, leave `cuda=None` and `device_map="auto"`
so both GPUs stay visible for sharding.
"""
import json
import math
import os
import pickle
from collections import defaultdict
from types import SimpleNamespace

# =============================================================================
# EDIT THESE FOR YOUR RUN (this replaces command-line args -- argparse reads
# sys.argv, which in a notebook cell is the kernel launcher's own args, not
# yours, so a plain config object is used instead).
# =============================================================================
args = SimpleNamespace(
    dataset_json="/kaggle/input/datasets/kmazd1110/late-alighnment-train-test-dataset/data/official_clevr_ladder/L1_train.json",  # path to L*_train.json / L*_val.json
    model_path="/kaggle/input/models/google/gemma-3/transformers/gemma-3-4b-it/1",
    cuda=None,          # e.g. "0" or "0,1" to restrict visible GPUs; None lets device_map="auto" see all of them
    device_map="cuda:0",  # a 4b model fits one T4 -- avoid the cross-GPU handoff overhead of sharding a model that doesn't need it.
                           # Only use "auto" once you switch to a 12b+ model that doesn't fit on a single 16GB card.
    dtype="bfloat16",   # Gemma's hidden states overflow float16's range (~65504) in mid/later layers -> NaN logits from
                         # that layer onward. bf16's wider exponent range avoids this. Don't switch to float16 for speed --
                         # the corpus subsampling + single-GPU device_map below already remove the real bottleneck.
    ps=True,            # do_pan_and_scan
    batch_size=2,       # each sample can expand to up to 5 sub-images (1 global + up to 4 PAS crops) through the vision
                         # tower, so this multiplies fast -- raise cautiously and only if you confirm free VRAM headroom.
    sample_size=200,    # -1 for full dataset
    attn_implementation="sdpa",  # sdpa/eager/flash_attention_2
    image_prefix_from="/kaggle/input",  # prefix in dataset_json's 'image' paths to replace (only used if image_prefix_to is set)
    image_prefix_to=None,  # local root to substitute for image_prefix_from, e.g. when the CLEVR image dataset mounts elsewhere
    corpus=None,        # path to a text corpus for latentlens to index; None = auto-use the bundled concepts.txt
    max_index_sentences=3000,  # subsample the corpus to this many sentences before indexing (concepts.txt has ~117k --
                                # massive overkill for an initial emergence probe over a handful of shape words). -1 = use all.
    index_cache_dir="clevr_ladder_results/latentlens_index",  # where the built ContextualIndex is saved/loaded from
    latentlens_top_k=5,
    index_batch_size=32,
)
# =============================================================================

if args.cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # reduce OOM from allocator fragmentation

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from tqdm import tqdm
import latentlens

DO_PAN_AND_SCAN = args.ps
DATASET_TAG = os.path.splitext(os.path.basename(args.dataset_json))[0]
OUTPUT_FILE_NAME = f"clevr_ladder_results/GEMMA/dataset{DATASET_TAG}_model{args.model_path.replace('/', '_')}"
os.makedirs(os.path.dirname(OUTPUT_FILE_NAME), exist_ok=True)

RESIZED_SIZE = 896
TOKENS_PER_SIDE = 16
VISION_START_ID = 255999

PROMPT_SUFFIX = "\nAnswer with yes or no."


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ClevrLadderDataset(Dataset):
    """Only keeps 'yes' items with a scene object matching target_word,
    since that's what gives us a bbox to localize the concept in."""

    def __init__(self, dataset_json, image_prefix_from=None, image_prefix_to=None):
        with open(dataset_json, "r") as f:
            raw = json.load(f)

        self.image_prefix_from = image_prefix_from
        self.image_prefix_to = image_prefix_to

        self.data = []
        for item in raw:
            answer = item["conversations"][1]["value"].strip().lower()
            if answer != "yes":
                continue
            target_word = item["target_word"].strip()
            match = next((o for o in item["scene_objects"] if o["shape"] == target_word), None)
            if match is None:
                continue
            self.data.append((item, match))

    def __len__(self):
        return len(self.data)

    def _resolve_image_path(self, path):
        if self.image_prefix_to is not None and path.startswith(self.image_prefix_from):
            return self.image_prefix_to + path[len(self.image_prefix_from):]
        return path

    def __getitem__(self, idx):
        item, obj = self.data[idx]
        image = Image.open(self._resolve_image_path(item["image"])).convert("RGB")
        question = item["conversations"][0]["value"].replace("<image>\n", "").strip()
        return {
            "id": item["id"],
            "image": image,
            "prompt": question + PROMPT_SUFFIX,
            "target_word": item["target_word"].strip(),
            "bbox": tuple(obj["bbox_2d"]),  # (x1, y1, x2, y2) in original pixel coords
        }


# ---------------------------------------------------------------------------
# Pan-and-scan / token-index helpers (identical to 2D_shape_recognition version)
# ---------------------------------------------------------------------------
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


def bbox_to_token_indices(x_min, y_min, x_max, y_max, original_width, original_height,
                           resized_size=RESIZED_SIZE, tokens_per_side=TOKENS_PER_SIDE):
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


def get_absolute_token_positions(token_indices, input_ids, image_index=0):
    image_starts = [i for i, x in enumerate(input_ids[0].tolist()) if x == VISION_START_ID]
    offset = image_starts[image_index] + 1
    return [i + offset for i in token_indices]


def find_tokens_in_crops(input_ids, bbox, crops, first_crop_input_ids_idx,
                          resized_size=RESIZED_SIZE, tokens_per_side=TOKENS_PER_SIDE):
    image_starts = [i for i, x in enumerate(input_ids[0].tolist()) if x == VISION_START_ID]
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


def get_region_token_indices(bbox, image, sample_input_ids, crops, first_crop_idx, global_img_idx):
    if DO_PAN_AND_SCAN and crops:
        return find_tokens_in_crops(sample_input_ids, bbox, crops, first_crop_idx)
    rel = bbox_to_token_indices(bbox[0], bbox[1], bbox[2], bbox[3], image.width, image.height)
    return get_absolute_token_positions(rel, sample_input_ids, image_index=global_img_idx)


def make_collate_fn(processor):
    def collate_fn(batch):
        images_list = [[item["image"]] for item in batch]
        messages_list = [
            [{"role": "user", "content": [
                {"type": "image", "image": item["image"]},
                {"type": "text", "text": item["prompt"]},
            ]}]
            for item in batch
        ]
        texts = processor.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True)
        inputs = processor(images_list, texts, padding=True, return_tensors="pt")
        return inputs, batch
    return collate_fn


# ---------------------------------------------------------------------------
# Synonym set per shape word. A model describing a photorealistic rendered
# object may route probability through a colloquial synonym ("ball", "box")
# rather than the literal geometric term used in CLEVR's own vocabulary --
# checking only the exact word can make real signal look like none at all.
# ---------------------------------------------------------------------------
SHAPE_SYNONYMS = {
    "sphere": ["sphere", "ball", "orb", "globe", "round"],
    "cube": ["cube", "box", "block", "square", "cubic"],
    "cylinder": ["cylinder", "tube", "pipe", "can", "roll"],
}


def get_synonym_token_ids(tokenizer, word):
    """Returns the set of single-token ids for `word` and its synonyms.
    Multi-token synonyms are skipped (with a one-time warning) since the
    per-patch probability metric only checks a single vocab id per word."""
    ids = set()
    for w in SHAPE_SYNONYMS.get(word, [word]):
        enc = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(enc[0])
        else:
            print(f"  (skipping synonym {w!r} for {word!r}: tokenizes to {len(enc)} pieces, not 1)")
    return ids


# ---------------------------------------------------------------------------
# Main probing loop
# ---------------------------------------------------------------------------
def find_bundled_corpus():
    """Locate concepts.txt inside the installed `latentlens` package (pip install latentlens
    already ships this file -- no need to git clone the repo just to get it)."""
    try:
        import importlib.resources as res
        for candidate in ("latentlens.data", "latentlens"):
            try:
                path = res.files(candidate).joinpath("concepts.txt")
                if path.is_file():
                    return str(path)
            except (ModuleNotFoundError, FileNotFoundError):
                continue
    except Exception:
        pass

    pkg_dir = os.path.dirname(latentlens.__file__)
    for candidate in (os.path.join(pkg_dir, "data", "concepts.txt"), os.path.join(pkg_dir, "concepts.txt")):
        if os.path.isfile(candidate):
            return candidate
    return None


def build_or_load_latentlens_index(model, tokenizer, layers):
    """Load a cached ContextualIndex if present, else build one from --corpus
    (or the corpus bundled inside the installed `latentlens` package) and cache it.
    Gemma-3 has no prebuilt index on the Hub, so this always builds locally the
    first time (one-off cost; reused on subsequent runs).
    """
    metadata_path = os.path.join(args.index_cache_dir, "metadata.json")
    if os.path.exists(metadata_path):
        print(f"Loading cached LatentLens index from {args.index_cache_dir}")
        return latentlens.ContextualIndex.from_directory(args.index_cache_dir, layers=layers)

    corpus = args.corpus or find_bundled_corpus()
    if corpus:
        print(f"Using corpus: {corpus}")
    args.corpus = corpus

    if not args.corpus:
        raise ValueError(
            "No cached LatentLens index found at --index_cache_dir, no --corpus was given, and the "
            "bundled concepts.txt couldn't be located inside the installed latentlens package. "
            "Pass --corpus pointing at a text corpus so the index can be built once."
        )

    with open(args.corpus, "r", encoding="utf-8") as f:
        corpus_lines = [line.strip() for line in f if line.strip()]

    if args.max_index_sentences != -1 and len(corpus_lines) > args.max_index_sentences:
        import random
        rng = random.Random(0)
        corpus_lines = rng.sample(corpus_lines, args.max_index_sentences)

    print(f"Building LatentLens index from {args.corpus} "
          f"({len(corpus_lines)} sentences after subsampling) for layers {layers} "
          f"(one-off, cached afterwards)...")
    index = latentlens.build_index(
        model_name=None,  # unused when `model=` is given directly
        model=model,
        tokenizer=tokenizer,
        corpus=corpus_lines,
        layers=layers,
        batch_size=args.index_batch_size,
    )
    os.makedirs(args.index_cache_dir, exist_ok=True)
    index.save(args.index_cache_dir)
    return index


def main():
    print(f"Using model: {args.model_path}")
    dataset = ClevrLadderDataset(args.dataset_json, args.image_prefix_from, args.image_prefix_to)
    print(f"{len(dataset)} usable ('yes' + localizable) examples out of the raw json")

    num_samples = len(dataset) if args.sample_size == -1 else min(args.sample_size, len(dataset))
    subset = Subset(dataset, range(num_samples))

    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True, padding_side="left", do_pan_and_scan=DO_PAN_AND_SCAN
    )
    torch_dtype = getattr(torch, args.dtype)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch_dtype, attn_implementation=args.attn_implementation,
        trust_remote_code=True, device_map=args.device_map,
    ).eval()
    print(f"Device map: {getattr(model, 'hf_device_map', model.device)}")

    N_LAYERS = len(model.model.language_model.layers) + 1
    latentlens_index = build_or_load_latentlens_index(model, processor.tokenizer, layers=list(range(N_LAYERS)))
    torch.cuda.empty_cache()  # release cached (but unused) allocator blocks from the text-only index build
    # before the image-heavy probing loop starts below.

    dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, collate_fn=make_collate_fn(processor))

    # per layer, list of per-sample scalars -- "_syn" variants also count synonyms (see SHAPE_SYNONYMS),
    # since a model may describe a rendered object as "ball"/"box"/"tube" rather than the literal geometric term.
    logit_lens_prob = defaultdict(list)         # mean prob mass on the LITERAL target token, over region patches
    logit_lens_hit_frac = defaultdict(list)     # fraction of region patches whose top-1 == literal target word
    logit_lens_prob_syn = defaultdict(list)     # mean prob mass summed over target word + its synonyms
    logit_lens_hit_frac_syn = defaultdict(list)  # fraction of region patches whose top-1 is in {target word, synonyms}
    latent_lens_hit_frac = defaultdict(list)    # fraction of region patches whose LatentLens neighbors mention the literal word
    latent_lens_hit_frac_syn = defaultdict(list)  # same, but counting any synonym as a hit too
    latent_lens_top1_sim = defaultdict(list)    # mean top-1 neighbor cosine similarity (LatentLens)

    all_data = []  # raw per-sample, per-layer records for later (finetune comparison) analysis

    for inputs, meta_batch in tqdm(dataloader):
        inputs = inputs.to(model.device)
        B = len(meta_batch)

        crops_info = []
        target_ids = []
        synonym_ids_list = []  # sorted list of vocab ids per sample: target word + its synonyms
        for item in meta_batch:
            crops = compute_pan_and_scan_crops(item["image"].width, item["image"].height) if DO_PAN_AND_SCAN else []
            global_img_idx = 0
            first_crop_idx = 1
            crops_info.append((crops, first_crop_idx, global_img_idx))

            word = item["target_word"].strip()
            tid = processor.tokenizer.encode(item["target_word"], add_special_tokens=False)[0]
            target_ids.append(tid)
            syn_ids = get_synonym_token_ids(processor.tokenizer, word) | {tid}
            synonym_ids_list.append(sorted(syn_ids))

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        if any(torch.isnan(hs).any() for hs in outputs.hidden_states):
            raise RuntimeError(
                "NaN detected in hidden states. If args.dtype == 'float16', this is almost certainly "
                "Gemma's activations overflowing float16's range in later layers -- switch args.dtype "
                "to 'bfloat16' (or 'float32') and rerun."
            )

        batch_sample_data = [[] for _ in range(B)]

        for layer in range(N_LAYERS):
            layer_hs = outputs.hidden_states[layer]  # [B, seq_len, D]

            for b_idx in range(B):
                item = meta_batch[b_idx]
                sample_hs = layer_hs[b_idx: b_idx + 1]
                sample_input_ids = inputs.input_ids[b_idx: b_idx + 1]
                crops, first_crop_idx, global_img_idx = crops_info[b_idx]
                target_id = target_ids[b_idx]
                synonym_ids = synonym_ids_list[b_idx]

                region_indices = get_region_token_indices(
                    item["bbox"], item["image"], sample_input_ids, crops, first_crop_idx, global_img_idx
                )
                region_features = sample_hs[0, region_indices, :]  # [n_patches, D]

                with torch.no_grad():
                    # --- logit lens ---
                    normalized = model.model.language_model.norm(region_features)
                    logits = model.lm_head(normalized)
                    probs = torch.softmax(logits, dim=-1)
                    top1_ids = torch.argmax(probs, dim=-1)

                    target_probs = probs[:, target_id]
                    hit_frac = (top1_ids == target_id).float().mean().item()

                    syn_ids_t = torch.tensor(synonym_ids, device=probs.device)
                    syn_probs = probs[:, syn_ids_t].sum(dim=-1)  # prob mass summed over word + its synonyms
                    hit_frac_syn = torch.isin(top1_ids, syn_ids_t).float().mean().item()

                # --- latent lens (McGill-NLP/latentlens) ---
                # Search each region patch's hidden state against the contextual index,
                # restricted to this layer, and check whether the retrieved neighbors
                # semantically mention the target word (or a synonym).
                # With device_map="auto" sharding the model across 2 GPUs, region_features
                # for a given layer may live on either card -- move to the index's device first.
                query = region_features.to(latentlens_index.device)
                neighbor_lists = latentlens_index.search(query, top_k=args.latentlens_top_k, layers=[layer])
                target_word_lc = item["target_word"].strip().lower()
                synonym_words_lc = [w.lower() for w in SHAPE_SYNONYMS.get(target_word_lc, [target_word_lc])]
                patch_hits, patch_hits_syn, patch_top1_sims = [], [], []
                for neighbors in neighbor_lists:
                    texts = [n.token_str.lower() for n in neighbors] + [(n.caption or "").lower() for n in neighbors]
                    hit = any(target_word_lc in t for t in texts)
                    hit_syn = hit or any(any(w in t for t in texts) for w in synonym_words_lc)
                    patch_hits.append(1.0 if hit else 0.0)
                    patch_hits_syn.append(1.0 if hit_syn else 0.0)
                    patch_top1_sims.append(neighbors[0].similarity if neighbors else 0.0)
                latent_hit_frac = float(np.mean(patch_hits)) if patch_hits else 0.0
                latent_hit_frac_syn = float(np.mean(patch_hits_syn)) if patch_hits_syn else 0.0
                latent_top1_sim = float(np.mean(patch_top1_sims)) if patch_top1_sims else 0.0

                mean_target_prob = target_probs.mean().item()
                mean_target_prob_syn = syn_probs.mean().item()

                logit_lens_prob[layer].append(mean_target_prob)
                logit_lens_hit_frac[layer].append(hit_frac)
                logit_lens_prob_syn[layer].append(mean_target_prob_syn)
                logit_lens_hit_frac_syn[layer].append(hit_frac_syn)
                latent_lens_hit_frac[layer].append(latent_hit_frac)
                latent_lens_hit_frac_syn[layer].append(latent_hit_frac_syn)
                latent_lens_top1_sim[layer].append(latent_top1_sim)

                batch_sample_data[b_idx].append({
                    "target_prob": mean_target_prob,
                    "hit_frac": hit_frac,
                    "target_prob_syn": mean_target_prob_syn,
                    "hit_frac_syn": hit_frac_syn,
                    "latent_hit_frac": latent_hit_frac,
                    "latent_hit_frac_syn": latent_hit_frac_syn,
                    "latent_top1_sim": latent_top1_sim,
                })

        for b_idx, item in enumerate(meta_batch):
            all_data.append({
                "id": item["id"],
                "target_word": item["target_word"],
                "per_layer": batch_sample_data[b_idx],
            })

    # -----------------------------------------------------------------
    # Aggregate + plot
    # -----------------------------------------------------------------
    layers = sorted(logit_lens_prob.keys())
    mean_prob = [np.mean(logit_lens_prob[l]) for l in layers]
    mean_hit_frac = [np.mean(logit_lens_hit_frac[l]) for l in layers]
    mean_prob_syn = [np.mean(logit_lens_prob_syn[l]) for l in layers]
    mean_hit_frac_syn = [np.mean(logit_lens_hit_frac_syn[l]) for l in layers]
    mean_latent_hit_frac = [np.mean(latent_lens_hit_frac[l]) for l in layers]
    mean_latent_hit_frac_syn = [np.mean(latent_lens_hit_frac_syn[l]) for l in layers]
    mean_latent_sim = [np.mean(latent_lens_top1_sim[l]) for l in layers]

    print("\nLayer | logit P(word) | logit P(word+syn) | logit hit | logit hit+syn | latent hit | latent hit+syn | latent sim")
    for l, p, ps, h, hs, lh, lhs, ls in zip(
        layers, mean_prob, mean_prob_syn, mean_hit_frac, mean_hit_frac_syn,
        mean_latent_hit_frac, mean_latent_hit_frac_syn, mean_latent_sim,
    ):
        print(f"{l:5d} | {p:.4f} | {ps:.4f} | {h:.4f} | {hs:.4f} | {lh:.4f} | {lhs:.4f} | {ls:.4f}")

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(layers, mean_prob, color="tab:blue", label="Logit lens: P(literal word)")
    ax1.plot(layers, mean_prob_syn, color="tab:blue", linestyle="--", label="Logit lens: P(word + synonyms)")
    ax1.plot(layers, mean_latent_hit_frac, color="tab:orange", label="LatentLens: neighbor hit (literal)")
    ax1.plot(layers, mean_latent_hit_frac_syn, color="tab:orange", linestyle="--", label="LatentLens: neighbor hit (+synonyms)")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Hit fraction / probability", color="tab:blue")
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, N_LAYERS)

    ax2 = ax1.twinx()
    ax2.plot(layers, mean_latent_sim, color="tab:red", label="LatentLens: mean top-1 neighbor similarity")
    ax2.set_ylabel("LatentLens top-1 similarity", color="tab:red")

    peak_prob_layer = int(layers[int(np.argmax(mean_prob_syn))])
    peak_latent_layer = int(layers[int(np.argmax(mean_latent_hit_frac_syn))])
    ax1.axvline(peak_prob_layer, color="tab:blue", alpha=0.3, linestyle=":")
    ax1.axvline(peak_latent_layer, color="tab:orange", alpha=0.3, linestyle=":")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)
    plt.title(f"Concept emergence across layers ({DATASET_TAG}, n={num_samples})\n"
              f"peak logit-lens (+syn) layer={peak_prob_layer}, peak latent-lens (+syn) layer={peak_latent_layer}")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_FILE_NAME}.png")

    with open(f"{OUTPUT_FILE_NAME}_all_data.pkl", "wb") as f:
        pickle.dump(all_data, f)
    with open(f"{OUTPUT_FILE_NAME}_layer_means.pkl", "wb") as f:
        pickle.dump({
            "layers": layers,
            "logit_lens_prob": mean_prob,
            "logit_lens_hit_frac": mean_hit_frac,
            "logit_lens_prob_syn": mean_prob_syn,
            "logit_lens_hit_frac_syn": mean_hit_frac_syn,
            "latent_lens_hit_frac": mean_latent_hit_frac,
            "latent_lens_hit_frac_syn": mean_latent_hit_frac_syn,
            "latent_lens_top1_sim": mean_latent_sim,
        }, f)

    print(f"\nSaved: {OUTPUT_FILE_NAME}.png / _all_data.pkl / _layer_means.pkl")


if __name__ == "__main__":
    main()

