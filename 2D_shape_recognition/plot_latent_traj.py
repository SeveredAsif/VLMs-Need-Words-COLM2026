###CELL 2
# ----------------------------------------------------------------------------
# Per-patch trajectory plots (reproduces logit_lens_traj.ipynb), generalized to
# work for either lens via the LENS toggle below. Paste into its own cell and
# run after Cell 1 (no GPU needed -- it only reads back the saved pickles).
# ----------------------------------------------------------------------------
import os
import json
import math
import pickle
import textwrap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from nltk.stem import PorterStemmer

LENS = "logit"          # "logit" or "latent"
DATASET_NAME = "basic_shapes_TEST"
MODEL_PATH = "google/gemma-3-12b-it"
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


patch_descriptions = describe_patches(DATASET_NAME, DATASET_ROOT_LOCAL, data_idx, option_idx, DO_PAN_AND_SCAN)
if patch_descriptions is not None and len(patch_descriptions) != len(lens_data[0]):
    print("[warning: recomputed patch count doesn't match saved data -- "
          "DO_PAN_AND_SCAN above may not match how Cell 1 was actually run]")
    patch_descriptions = None


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


# Latent-lens labels can be whole corpus sentences (not single words), so they
# need a smaller font, a wider figure, and wrapped/truncated annotation text to
# stay readable -- logit-lens labels are always single vocabulary tokens.
IS_SENTENCE_LENS = (LENS == "latent")

FONTSIZE = 24 if not IS_SENTENCE_LENS else 13
TITLE_FONTSIZE = FONTSIZE + 6
LABEL_FONTSIZE = FONTSIZE + (0 if not IS_SENTENCE_LENS else 6)
TICK_FONTSIZE = FONTSIZE + (0 if not IS_SENTENCE_LENS else 6)
ANNOT_FONTSIZE = FONTSIZE
LEGEND_FONTSIZE = FONTSIZE
ANNOT_WRAP_WIDTH = 26       # chars per line inside an annotation bubble
ANNOT_MAX_CHARS = 110       # truncate very long corpus sentences beyond this

_stemmer = PorterStemmer()


def stem(word):
    return _stemmer.stem(word.lower()) if word else ""


def annotation_text(word):
    """Display text for an annotation bubble: word/token as-is for logit lens;
    wrapped + truncated for latent lens, since a label there can be a full
    corpus sentence instead of a single word."""
    if not IS_SENTENCE_LENS:
        return word
    text = word if len(word) <= ANNOT_MAX_CHARS else word[:ANNOT_MAX_CHARS - 3] + "..."
    return textwrap.fill(text, width=ANNOT_WRAP_WIDTH)


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

    fig, ax = plt.subplots(figsize=(16, 7) if not IS_SENTENCE_LENS else (22, 10))

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
                annotation_text(word), xy=(layer, value), xytext=(0, v_loc), textcoords="offset points",
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
    if patch_descriptions is not None:
        where = patch_descriptions[target_token_idx]
        subtitle = f"image-grid cell: {where} of option {option_letter}'s box in the target image"
    else:
        subtitle = (f"{target_token_idx}-th image-grid cell (raster order) inside option "
                    f"{option_letter}'s box in the target image")

    ax.set(
        xlabel="Layer",
        ylabel=VALUE_LABEL,
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
    plt.ylim(-0.5, 1.50)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/patch{target_token_idx}.pdf", dpi=100, bbox_inches="tight")
    plt.close()

print(f"Saved {LENS}-lens trajectory plots to {save_dir}/")
