from collections import defaultdict

import numpy as np
import webcolors


def jaccard_similarity(set1, set2):
    s1, s2 = set(set1), set(set2)
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 0.0


def is_color(name):
    try:
        webcolors.name_to_hex(name)
        return True
    except ValueError:
        return False


def filter_tokens(
    tokens, ignore_colors=False, ignore_options=False, extra_color_tokens=frozenset()
):
    result = []
    for t in tokens:
        t_stripped = t.strip()
        if ignore_colors and is_color(t_stripped.lower()):
            continue
        if ignore_colors and t_stripped in extra_color_tokens:
            continue
        if ignore_options and t_stripped in {"A", "B", "C", "D"}:
            continue
        result.append(t)
    return result


def pairwise_jaccard_for_bboxes(
    all_tokens,
    num_bboxes=4,
    ignore_colors=False,
    ignore_options=False,
    extra_color_tokens=frozenset(),
):
    filtered = [
        filter_tokens(
            toks,
            ignore_colors=ignore_colors,
            ignore_options=ignore_options,
            extra_color_tokens=extra_color_tokens,
        )
        for toks in all_tokens
    ]
    return [
        jaccard_similarity(filtered[i], filtered[j])
        for i in range(num_bboxes)
        for j in range(i + 1, num_bboxes)
    ]


def compute_jac_scores(
    all_data, ignore_colors=False, ignore_options=False, extra_color_tokens=frozenset()
):
    per_layer_jac_scores = defaultdict(list)
    for all_shapes in all_data:
        for layer, shapes in enumerate(all_shapes):
            all_tokens = [[x[0] for x in shape] for shape in shapes]
            jac_scores = pairwise_jaccard_for_bboxes(
                all_tokens,
                ignore_colors=ignore_colors,
                ignore_options=ignore_options,
                extra_color_tokens=extra_color_tokens,
            )
            per_layer_jac_scores[layer].append(np.mean(jac_scores))
    return per_layer_jac_scores


def mean_jaccard_per_layer(per_layer_jac_scores):
    return [
        [layer, np.mean(scores)]
        for layer, scores in per_layer_jac_scores.items()
    ]
