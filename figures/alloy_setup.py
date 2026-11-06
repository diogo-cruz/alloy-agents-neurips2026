"""Figure 1: alloy setup (left) and the single-action vs continuous-effort asymmetry (right).

Replaces the submitted alloy_setup.png, whose continuous-effort panel was illustrative and
labelled "alloy is more dangerous". Every number here comes from the paper's tables:
  top right    scheming evals (tab:scheming_expanded): GPT 0/21, Gemini 4/15, alloy 1/15
  bottom right Bash, fresh runs on the first 100 tasks, Aug 2026 (reports/phase2_run.json):
               GPT main 84/96 side 0/96 both 0/96; Gemini 53/98, 94/98, 51/98;
               alloy 78/97, 66/97, 56/97
Whiskers are 95% Wilson intervals. Palette checked with the dataviz validator (light mode,
all pairs): the submitted alloy red #c53030 sat 13.1 dE from Gemini orange, below the
normal-vision floor of 15, so the alloy uses #a61e4d here (worst pair 16.4).

Run from figures/: python alloy_setup.py  (writes alloy_setup.pdf)
"""
import math

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

plt.rcParams.update({
    # Computer Modern via Matplotlib's bundled fonts; usetex would need cm-super on this box
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True,
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8,
})

GPT, GEM, ALLOY = "#2b6cb0", "#dd6b20", "#a61e4d"
INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=2)
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def bars(ax, xs, counts, colors, width):
    for x, (k, n), c in zip(xs, counts, colors):
        v = 100 * k / n
        lo, hi = wilson(k, n)
        ax.bar(x, v, width, color=c, edgecolor="white", linewidth=1.0, zorder=2)
        ax.plot([x, x], [lo, hi], color=INK, lw=0.7, zorder=3)
        ax.text(x, hi + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK)


fig = plt.figure(figsize=(6.9, 2.75))
gs = fig.add_gridspec(2, 2, width_ratios=[0.95, 1.35], height_ratios=[1, 1.25],
                      wspace=0.22, hspace=0.62, left=0.01, right=0.995, top=0.83, bottom=0.11)

# ---- left: schematic ------------------------------------------------------------
ax = fig.add_subplot(gs[:, 0])
ax.set_xlim(0, 3.3)
ax.set_ylim(-0.75, 6.0)
ax.axis("off")
cols = [("$A$-solo", ["A"] * 5), ("$B$-solo", ["B"] * 5), ("$A{+}B$ alloy", ["A", "B"] * 2 + ["A"])]
for ci, (label, seq) in enumerate(cols):
    x0 = 0.45 + ci * 0.98
    ax.text(x0 + 0.33, 5.55, label, ha="center", va="center", fontsize=9, color=INK,
            fontweight="bold" if ci == 2 else "normal")
    for ri, who in enumerate(seq):
        y = 4.55 - ri * 0.98
        c = GPT if who == "A" else GEM
        ax.add_patch(FancyBboxPatch((x0, y), 0.66, 0.66, boxstyle="round,pad=0.0,rounding_size=0.09",
                                    fc=c, ec="white", lw=1.0))
        ax.text(x0 + 0.33, y + 0.33, f"${who}$", ha="center", va="center", color="white", fontsize=9)
ax.add_patch(FancyBboxPatch((0.45 + 2 * 0.98 - 0.09, 0.1 - 0.09), 0.84, 5.18,
                            boxstyle="round,pad=0.0,rounding_size=0.12",
                            fc="none", ec=MUTED, lw=0.8, ls=(0, (3, 2))))
ax.annotate("", xy=(0.2, 0.25), xytext=(0.2, 5.1),
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8))
ax.text(0.08, 2.68, "turns", rotation=90, ha="center", va="center", fontsize=8, color=MUTED)
ax.text(1.75, -0.42, "one shared transcript per run; each model sees the\nfull history but not who wrote earlier turns",
        ha="center", va="center", fontsize=7, color=MUTED, linespacing=1.15)

# ---- top right: single-action -----------------------------------------------------
ax1 = fig.add_subplot(gs[0, 1])
style(ax1)
bars(ax1, [0, 1, 2], [(0, 21), (4, 15), (1, 15)], [GPT, GEM, ALLOY], 0.5)
ax1.set_xticks([0, 1, 2], ["GPT-5-mini", "Gemini", "Alloy"])
ax1.set_ylim(0, 75)
ax1.set_yticks([0, 25, 50])
ax1.set_ylabel("Unsafe (%)")
ax1.set_title("Single-action (scheming evals)", loc="left", color=INK)

# ---- bottom right: continuous-effort composition -----------------------------------
ax2 = fig.add_subplot(gs[1, 1])
style(ax2)
metrics = ["Main task", "Side task", "Both in one run"]
data = {GPT: [(84, 96), (0, 96), (0, 96)],
        GEM: [(53, 98), (94, 98), (51, 98)],
        ALLOY: [(78, 97), (66, 97), (56, 97)]}
w = 0.25
for j, c in enumerate((GPT, GEM, ALLOY)):
    bars(ax2, [i + (j - 1) * (w + 0.02) for i in range(3)], data[c], [c] * 3, w)
ax2.set_xticks(range(3), metrics)
ax2.set_ylim(0, 118)
ax2.set_yticks([0, 50, 100])
ax2.set_ylabel("Runs (%)")
ax2.set_title("Continuous-effort (Bash, fresh runs on 100 tasks)", loc="left", color=INK)

fig.legend(handles=[Patch(fc=GPT, label="GPT-5-mini solo"), Patch(fc=GEM, label="Gemini 3.1 Flash Lite solo"),
                    Patch(fc=ALLOY, label="GPT+Gemini alloy")],
           loc="upper right", ncol=3, frameon=False, fontsize=7.5, handlelength=1.0,
           columnspacing=1.2, bbox_to_anchor=(0.995, 1.0))
fig.savefig("alloy_setup.pdf")
print("[saved] alloy_setup.pdf")
