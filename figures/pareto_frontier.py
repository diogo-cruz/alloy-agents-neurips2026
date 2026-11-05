"""Generate a 1×3 Pareto frontier figure with SEM error bars."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


plt.rcParams.update(
    {
        # Computer Modern via Matplotlib's bundled fonts; usetex would need cm-super here
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    }
)


COLORS = {
    "GPT": "#2b6cb0",
    "Gemini": "#dd6b20",
    "Grok": "#38a169",
    "Alloy GPT+Gemini": "#a61e4d",  # matches Figs 1-2; #c53030 failed the validator vs Gemini orange
    "Alloy GPT+Grok": "#805ad5",
    "Alloy Gemini+Grok": "#744210",  # was #9b2c2c, 8.8 dE from the GPT+Gemini red
}

MARKERS = {
    "GPT": "o",
    "Gemini": "s",
    "Grok": "^",
    "Alloy GPT+Gemini": "D",
    "Alloy GPT+Grok": "P",
    "Alloy Gemini+Grok": "X",
}


def binomial_sem(k: int, n: int) -> float:
    """Standard error of the proportion, in percent."""
    if n == 0:
        return 0.0
    p = k / n
    return math.sqrt(p * (1 - p) / n) * 100


# Bash and iac_fast: (k, n) tuples for binomial SEM
# bash_arena: "mean" type with (mean_pct, se_pct) for main axis
PANELS = [
    {
        "title": "Bash",
        "points": {
            "GPT":                {"main": (21, 21), "side": (0, 21)},
            "Gemini":             {"main": (69, 143), "side": (136, 143)},
            "Grok":               {"main": (16, 21), "side": (15, 21)},
            "Alloy GPT+Gemini":   {"main": (43, 49), "side": (39, 49)},
            "Alloy GPT+Grok":     {"main": (18, 19), "side": (8, 19)},
            "Alloy Gemini+Grok":  {"main": (16, 20), "side": (20, 20)},
        },
    },
    {
        "title": r"$\mathrm{iac\_fast}$",  # cmr10 has no underscore glyph; mathtext does
        "points": {
            "GPT":                {"main": (19, 25), "side": (9, 25)},
            "Gemini":             {"main": (61, 105), "side": (48, 105)},
            "Grok":               {"main": (9, 25),  "side": (8, 25)},
            "Alloy GPT+Gemini":   {"main": (54, 80), "side": (38, 80)},
            "Alloy GPT+Grok":     {"main": (18, 44), "side": (17, 44)},
            "Alloy Gemini+Grok":  {"main": (18, 45), "side": (23, 45)},
        },
    },
    {
        "title": "Bash Arena",
        # main: (mean_pct, se_pct) — mean fractional scores, SE already provided
        # side: (k, n) — binary, uses binomial SEM
        "main_type": "mean",
        "points": {
            "GPT":                {"main": (42.3, 7.4), "side": (9, 17)},
            "Gemini":             {"main": (20.5, 2.7), "side": (66, 75)},
            "Grok":               {"main": (33.4, 11.1), "side": (6, 7)},
            "Alloy GPT+Gemini":   {"main": (39.8, 3.2), "side": (45, 64)},  # canonical pool (was 73 runs incl. an ambiguous 9-run file)
            "Alloy GPT+Grok":     {"main": (34.8, 4.7), "side": (37, 40)},
            "Alloy Gemini+Grok":  {"main": (30.3, 4.2), "side": (38, 40)},
        },
    },
]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.8), sharey=True)

    for ax, panel in zip(axes, PANELS):
        ax.grid(True, color="#d1d5db", lw=0.8, alpha=0.9, zorder=0)
        main_type = panel.get("main_type", "binomial")

        for label, values in panel["points"].items():
            # Side axis is always binomial (k, n)
            side_k, side_n = values["side"]
            y = 100 * side_k / side_n
            y_sem = binomial_sem(side_k, side_n)

            # Main axis: binomial or mean
            if main_type == "mean":
                x = values["main"][0]  # mean_pct
                x_sem = values["main"][1]  # se_pct (already SEM)
            else:
                main_k, main_n = values["main"]
                x = 100 * main_k / main_n
                x_sem = binomial_sem(main_k, main_n)

            is_alloy = "Alloy" in label
            ax.errorbar(
                x,
                y,
                xerr=x_sem,
                yerr=y_sem,
                fmt="none",
                ecolor=COLORS[label],
                elinewidth=1.6,
                capsize=3.0,
                alpha=0.9,
                zorder=2,
            )
            ax.scatter(
                x,
                y,
                s=60 if is_alloy else 45,
                marker=MARKERS[label],
                color=COLORS[label],
                edgecolors="white",
                linewidths=1.0,
                zorder=3,
            )

        ax.set_title(panel["title"], fontweight="bold", pad=8)
        ax.set_xlim(0, 102)
        ax.set_ylim(0, 102)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)

    axes[0].set_ylabel("Side-task compliance (%)", fontsize=11)
    fig.supxlabel("Main-task success (%)", fontsize=11, y=0.08)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["GPT"], markeredgecolor="white", markersize=8, label="GPT solo"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["Gemini"], markeredgecolor="white", markersize=8, label="Gemini solo"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=COLORS["Grok"], markeredgecolor="white", markersize=8, label="Grok solo"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["Alloy GPT+Gemini"], markeredgecolor="white", markersize=8, label="Alloy GPT+Gem."),
        Line2D([0], [0], marker="P", color="none", markerfacecolor=COLORS["Alloy GPT+Grok"], markeredgecolor="white", markersize=8, label="Alloy GPT+Grok"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=COLORS["Alloy Gemini+Grok"], markeredgecolor="white", markersize=8, label="Alloy Gem.+Grok"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper center",
        bbox_to_anchor=(0.5, -0.04), ncol=6,
        fontsize=8, frameon=False, columnspacing=1.0,
    )

    fig.tight_layout(rect=[0, 0.10, 1, 1], w_pad=1.0)
    plt.savefig("pareto_frontier.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.savefig("pareto_frontier.png", bbox_inches="tight", pad_inches=0.02, dpi=260)
    print("Saved pareto_frontier.pdf and pareto_frontier.png")


if __name__ == "__main__":
    main()
