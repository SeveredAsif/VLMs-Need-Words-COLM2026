import matplotlib as mpl
import matplotlib.pyplot as plt

_COLORS = ["#0072B2", "#D55E00", "#009E73"]
_MARKERS = ["o", "s", "^"]


def configure_plot_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "gray",
            "axes.linewidth": 1.2,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
        }
    )


def save_plot(v1, v2, v1_name, v2_name, filename, v3=None, v3_name=None):
    v1_layers, v1_scores = zip(*v1)
    v2_layers, v2_scores = zip(*v2)

    _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        v1_layers,
        v1_scores,
        marker=_MARKERS[0],
        label=v1_name,
        linewidth=2,
        markersize=5,
        color=_COLORS[0],
    )
    ax.plot(
        v2_layers,
        v2_scores,
        marker=_MARKERS[1],
        label=v2_name,
        linewidth=2,
        markersize=5,
        color=_COLORS[1],
    )
    if v3 is not None:
        v3_layers, v3_scores = zip(*v3)
        ax.plot(
            v3_layers,
            v3_scores,
            marker=_MARKERS[2],
            label=v3_name,
            linewidth=2,
            markersize=5,
            color=_COLORS[2],
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Jaccard Similarity")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
