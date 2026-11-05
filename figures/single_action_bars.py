"""Generate a grouped bar chart of unsafe rates for single-action benchmarks.

Bars: Gemini, GPT, Alloy. Alloy bar has a hatched extension showing
the gap to the average of both solos (k=1 pooled). A diamond marker
shows the "best of 2" rate (run both, pick safest → unsafe only if
both unsafe → p_GPT × p_Gemini).

Benchmarks ordered from highest to lowest average unsafe rate.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
    "Alloy": "#a61e4d",  # was #c53030: 13.1 dE from Gemini orange, below the 15 floor
}


def binomial_sem(k: int, n: int) -> float:
    """Standard error of the mean for a binomial proportion, in percent."""
    if n == 0:
        return 0.0
    p = k / n
    return math.sqrt(p * (1 - p) / n) * 100


# Data: (k_unsafe, n_total) for each benchmark
# Ordered from highest to lowest average unsafe rate
BENCHMARKS = [
    {
        "name": "Impossible\nBench",
        "GPT": (32, 40),
        "Gemini": (6, 31),
        "Alloy": (18, 39),
    },
    {
        "name": "AgentDojo",
        "GPT": (2, 10),
        "Gemini": (6, 10),
        "Alloy": (2, 10),
    },
    {
        "name": "Data\nPoisoning",
        "GPT": (4, 10),
        "Gemini": (4, 10),
        "Alloy": (2, 10),
    },
    {
        "name": "Propensity\nBench",
        "GPT": (1, 36),
        "Gemini": (9, 30),
        "Alloy": (3, 30),
    },
    {
        "name": "Scheming",
        "GPT": (0, 21),
        "Gemini": (4, 15),
        "Alloy": (1, 15),
    },
]


def main():
    n_benchmarks = len(BENCHMARKS)
    x = np.arange(n_benchmarks)
    width = 0.25
    # Order: Gemini first (higher unsafe on average), then GPT, then Alloy
    configs = ["Gemini", "GPT", "Alloy"]
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(6.8, 2.4))

    # Track alloy bar positions and rates for overlays
    alloy_positions = []
    alloy_rates = []
    avg_rates = []
    best_of_2_rates = []

    for i, config in enumerate(configs):
        rates = []
        yerr_lo = []
        yerr_hi = []
        for b in BENCHMARKS:
            k, n = b[config]
            rate = 100 * k / n if n > 0 else 0
            sem = binomial_sem(k, n)
            rates.append(rate)
            yerr_lo.append(sem)
            yerr_hi.append(sem)

        label = f"{config} solo" if config != "Alloy" else "Alloy GPT+Gemini"
        bars = ax.bar(
            x + offsets[i],
            rates,
            width,
            label=label,
            color=COLORS[config],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.errorbar(
            x + offsets[i],
            rates,
            yerr=[yerr_lo, yerr_hi],
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=3,
            zorder=4,
        )

        if config == "Alloy":
            alloy_positions = list(x + offsets[i])
            alloy_rates = list(rates)

    # Compute average and best-of-2 rates
    for b in BENCHMARKS:
        k_gpt, n_gpt = b["GPT"]
        k_gem, n_gem = b["Gemini"]
        p_gpt = k_gpt / n_gpt if n_gpt > 0 else 0
        p_gem = k_gem / n_gem if n_gem > 0 else 0
        avg_rates.append((p_gpt + p_gem) / 2 * 100)
        best_of_2_rates.append(p_gpt * p_gem * 100)

    # Hatched overlay: from alloy rate to average rate
    for j in range(n_benchmarks):
        gap = avg_rates[j] - alloy_rates[j]
        if gap > 0:
            ax.bar(
                alloy_positions[j],
                gap,
                width,
                bottom=alloy_rates[j],
                color="none",
                edgecolor=COLORS["Alloy"],
                hatch="///",
                linewidth=0.5,
                zorder=3,
                alpha=0.7,
            )

    # Best-of-2 markers (diamond)
    ax.scatter(
        alloy_positions,
        best_of_2_rates,
        marker="D",
        s=28,
        color="black",
        zorder=5,
    )

    ax.set_ylabel("Unsafe behavior rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([b["name"] for b in BENCHMARKS], fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])

    # Custom legend
    legend_handles = [
        Patch(facecolor=COLORS["Gemini"], edgecolor="white", label="Gemini solo"),
        Patch(facecolor=COLORS["GPT"], edgecolor="white", label="GPT solo"),
        Patch(facecolor=COLORS["Alloy"], edgecolor="white", label="Alloy GPT+Gemini"),
        Patch(facecolor="none", edgecolor=COLORS["Alloy"], hatch="///",
              label="Gap to solo average", alpha=0.7),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="black",
               markersize=5, label="Best of 2 (run both, pick safest)"),
    ]
    ax.legend(
        handles=legend_handles, fontsize=9, loc="upper right",
        frameon=False, handlelength=1.8,
    )

    ax.grid(True, axis="y", color="#d1d5db", lw=0.8, alpha=0.9, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    fig.tight_layout()
    plt.savefig("single_action_bars.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.savefig("single_action_bars.png", bbox_inches="tight", pad_inches=0.02, dpi=260)
    print("Saved single_action_bars.pdf and single_action_bars.png")


if __name__ == "__main__":
    main()
