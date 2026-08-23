"""
Recompute the Jaccard-distinctiveness metric for logit lens AND latent lens
from the already-saved *_all_data.pkl files -- no GPU / model rerun needed,
since all_data already holds the decoded (label, value) per patch per layer.

This adds one extra filter on top of the existing color/option-letter filter:
drop special/control tokens (<bos>, <eos>, <pad>, ...) and pure-punctuation
tokens (e.g. "'", ".") before computing Jaccard. These dominate the LatentLens
readout at many layers (<bos> matches almost every patch in the middle/late
layers) and otherwise make the metric look artificially "distinguishable" or
"indistinguishable" for reasons unrelated to shape identity.

Reads/writes entirely within this repo's 2D_shape_recognition/logit_latent_results/.
"""
import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt

DATASET_NAME = "basic_shapes_TEST"
MODEL_PATH = "google/gemma-3-12b-it"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logit_latent_results")

IGNORE_COLORS = True
IGNORE_OPTIONS = True
IGNORE_SPECIAL = True   # the new filter -- see module docstring

try:
    import webcolors

    def is_color(name):
        try:
            webcolors.name_to_hex(name)
            return True
        except ValueError:
            return False
except ImportError:
    def is_color(name):
        return False


def is_special_or_boilerplate(t_stripped):
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
    """None (not 0.0) when both sets are empty after filtering -- e.g. every
    patch in both options decoded to <bos>/punctuation and got dropped. That
    means "no information survived filtering", the opposite of "distinguishable
    (Jaccard=0)", so it must NOT be folded into the mean as a 0."""
    s1, s2 = set(set1), set(set2)
    if not s1 and not s2:
        return None
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 0.0


def recompute(lens_name):
    file_path = f"{RESULTS_DIR}/dataset{DATASET_NAME}_model{MODEL_PATH.replace('/', '_')}_{lens_name}_all_data.pkl"
    with open(file_path, "rb") as f:
        all_data = pickle.load(f)  # all_data[data_idx][layer][option_idx] = [(label, value), ...]

    num_layers = len(all_data[0])
    per_layer_jac = {layer: [] for layer in range(num_layers)}
    per_layer_valid_frac = {layer: [] for layer in range(num_layers)}

    for sample in all_data:
        for layer in range(num_layers):
            option_tokens = [[label for label, _ in sample[layer][k]] for k in range(4)]
            filtered = [filter_tokens(toks) for toks in option_tokens]
            pair_scores = [jaccard_similarity(filtered[i], filtered[j])
                           for i in range(4) for j in range(i + 1, 4)]
            valid = [s for s in pair_scores if s is not None]
            per_layer_jac[layer].extend(valid)
            per_layer_valid_frac[layer].append(len(valid) / len(pair_scores))

    layers_sorted = sorted(per_layer_jac)
    means = [float(np.mean(per_layer_jac[layer])) if per_layer_jac[layer] else float("nan")
             for layer in layers_sorted]
    valid_fracs = [float(np.mean(per_layer_valid_frac[layer])) for layer in layers_sorted]
    return layers_sorted, means, valid_fracs


def plot_jaccard(layers, scores, valid_frac, label, color, filename, n_layers, low_valid_thresh=0.2):
    scores_arr = np.array(scores, dtype=float)
    plt.figure()
    plt.plot(layers, scores_arr, color=color)

    # Grey out layers where most/all option-pairs had nothing left after
    # filtering -- the mean there is noisy or (if all-NaN) simply undefined.
    low_valid = np.array(valid_frac) < low_valid_thresh
    for layer, is_low in zip(layers, low_valid):
        if is_low:
            plt.axvspan(layer - 0.5, layer + 0.5, color="grey", alpha=0.15, linewidth=0)

    plt.xlabel("Layer")
    plt.ylabel("Mean Jaccard Similarity")
    plt.xlim(0, n_layers)
    plt.ylim(0, 1)
    if np.any(~np.isnan(scores_arr)):
        max_score = float(np.nanmax(scores_arr))
        max_score_layer = layers[int(np.nanargmax(scores_arr))]
        plt.axhline(y=max_score, alpha=0.5, color='red', linestyle='--',
                    label=f'Max Jaccard: {max_score:.4f} at Layer {max_score_layer}')
    plt.plot([], [], color="grey", alpha=0.3, linewidth=8,
             label=f"<{low_valid_thresh*100:.0f}% of pairs had data (mostly <bos>-filtered)")
    plt.title(f"{label} (special/punctuation tokens filtered) -- lower = more distinguishable")
    plt.legend(fontsize=8)
    plt.savefig(filename)
    plt.close()


layers_logit, logit_means, logit_valid = recompute("logit")
layers_latent, latent_means, latent_valid = recompute("latent")
assert layers_logit == layers_latent
layers_sorted = layers_logit
N_LAYERS = len(layers_sorted)

print("\nMean Jaccard similarity per layer, special-token-filtered (logit lens vs. latent lens).")
print("'valid %' = fraction of the 6 option-pairs per sample that had >=1 surviving token on")
print("either side after filtering -- a low % means the mean at that layer is based on very")
print("little data (most patches decoded to <bos>/punctuation and got dropped entirely).")
print(f"{'Layer':>6} | {'logit lens':>12} | {'valid %':>8} | {'latent lens':>12} | {'valid %':>8}")
for layer, lg, lgv, lt, ltv in zip(layers_sorted, logit_means, logit_valid, latent_means, latent_valid):
    print(f"{layer:>6} | {lg:>12.4f} | {lgv*100:>7.1f}% | {lt:>12.4f} | {ltv*100:>7.1f}%")

OUT = f"{RESULTS_DIR}/dataset{DATASET_NAME}_model{MODEL_PATH.replace('/', '_')}_filtered"

plot_jaccard(layers_sorted, logit_means, logit_valid, "Logit Lens", "tab:blue", f"{OUT}_logit_jaccard.png", N_LAYERS)
plot_jaccard(layers_sorted, latent_means, latent_valid, "Latent Lens", "tab:orange", f"{OUT}_latent_jaccard.png", N_LAYERS)

plt.figure(figsize=(9, 5))
plt.plot(layers_sorted, logit_means, label="Logit Lens", color="tab:blue")
plt.plot(layers_sorted, latent_means, label="Latent Lens", color="tab:orange")
low_valid_latent = np.array(latent_valid) < 0.2
for layer, is_low in zip(layers_sorted, low_valid_latent):
    if is_low:
        plt.axvspan(layer - 0.5, layer + 0.5, color="grey", alpha=0.15, linewidth=0)
plt.plot([], [], color="grey", alpha=0.3, linewidth=8, label="latent lens: <20% of pairs had data")
plt.xlabel("Layer")
plt.ylabel("Mean Jaccard Similarity (lower = more distinguishable)")
plt.xlim(0, N_LAYERS)
plt.ylim(0, 1)
plt.title("Logit Lens vs. Latent Lens -- per-layer comparison (special tokens filtered)")
plt.legend(fontsize=8)
plt.savefig(f"{OUT}_comparison.png")
plt.close()

with open(f"{OUT}_logit_jaccard.pkl", "wb") as f:
    pickle.dump(logit_means, f)
with open(f"{OUT}_latent_jaccard.pkl", "wb") as f:
    pickle.dump(latent_means, f)

print(f"\nSaved filtered results under: {OUT}_*")
