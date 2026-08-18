"""Regenerate the three figures used in the DynaPersona manuscript.

Outputs PNG and SVG files into ./figures.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 11})


def save_both(fig, stem):
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


# Fig. 1: context-gated mixture architecture
fig, ax = plt.subplots(figsize=(4.6, 4.0))
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)


def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.03,rounding_size=0.08",
                                fc=fc, ec="#274b7a", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.1, color="#333333"))

box(2.1, 7.8, 5.8, 1.15, "Context $c$\n(persona, emotion, style slot)", "#fdefe6")
box(2.7, 5.65, 4.6, 1.05, "Gate  $g(c)$ = softmax", "#eaf6ee")
box(2.0, 3.35, 6.0, 1.10, "$K$ experts  $B_kA_k$\n(each rank $r/K$)", "#eef3fb")
box(2.0, 1.05, 6.0, 1.10, "Frozen $W_0 + \\Sigma_k g_k B_kA_k$", "#f2f2f2")
arrow(5.0, 7.8, 5.0, 6.7)
arrow(5.0, 5.65, 5.0, 4.45)
arrow(5.0, 3.35, 5.0, 2.15)
save_both(fig, "fig1_architecture")

# Fig. 2: perplexity comparison
models = ["Static\nLoRA", "SLIM\nrouted", "DynaPersona\n-MoE"]
means = [14.278, 14.145, 13.941]
stds = [0.038, 0.026, 0.012]
fig, ax = plt.subplots(figsize=(3.2, 3.0))
bars = ax.bar(models, means, yerr=stds, capsize=4, width=0.62)
ax.set_ylabel("Test perplexity")
ax.set_ylim(13.7, 14.45)
for b, m in zip(bars, means):
    ax.text(b.get_x() + b.get_width() / 2, m + 0.012, f"{m:.3f}", ha="center", fontsize=9)
ax.tick_params(labelsize=9)
save_both(fig, "fig2_perplexity")

# Fig. 3: routing weights grouped by source (representative seed 42).
# The historical result field is named `expert_usage_by_style`, but its two groups are the
# stored source labels. Both groups map to the same encoded style slot in the reported run.
empathetic = [0.308, 0.272, 0.162, 0.258]
persona = [0.175, 0.250, 0.339, 0.235]
x = np.arange(4)
width = 0.38
fig, ax = plt.subplots(figsize=(3.6, 3.0))
ax.bar(x - width / 2, empathetic, width, label="EmpatheticDialogues", hatch="///", edgecolor="black")
ax.bar(x + width / 2, persona, width, label="PersonaChat", hatch="...", edgecolor="black")
ax.axhline(0.25, ls="--", lw=0.8, color="#777777")
ax.set_xticks(x)
ax.set_xticklabels([f"E{i + 1}" for i in range(4)], fontsize=9)
ax.set_ylabel("Mean routing weight")
ax.set_xlabel("Expert")
ax.legend(fontsize=8, frameon=False)
ax.tick_params(labelsize=9)
save_both(fig, "fig3_expert_usage")

print("Wrote figures to", OUT)
