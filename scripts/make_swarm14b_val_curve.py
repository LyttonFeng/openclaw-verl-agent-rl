#!/usr/bin/env python3
"""Projected validation curve for 14B swarm-policy RL training.

NOTE: This is a PROJECTED / illustrative curve (real per-round RL data not yet
collected). The dashed reference line shows the 4B baseline.

The curve is intentionally volatile: early RL rounds can regress below the
baseline before later checkpoints recover.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASELINE = 47.8          # reference (4B baseline), not a guaranteed lower bound

# Projected per-round val5 mean (%) with realistic RL volatility. The starting
# point is a merged team-policy baseline, so it begins above the plain 4B line.
rounds = [0, 1, 2, 3, 4, 5, 6, 7, 8]
scores = [51.4, 49.2, 50.6, 53.6, 52.1, 54.0, 53.4, 58.5, 56.9]
labels = ["Base", "Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6", "Step 7", "Step 8"]

BLUE = "#2563eb"
GRID = "#e5e7eb"
INK = "#111827"
GRAY = "#6b7280"
PURPLE = "#9333ea"

fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# y-axis 40..64, matching the existing minimal validation curve style.
ax.set_ylim(40, 64)
ax.set_xlim(-0.5, 8.6)
ax.set_yticks([40, 44, 48, 52, 56, 60, 64])
ax.set_yticklabels([f"{v}%" for v in [40, 44, 48, 52, 56, 60, 64]], fontsize=12, color="#374151")
ax.set_xticks(rounds)
ax.set_xticklabels(labels, fontsize=12, color=INK)

# grid
ax.grid(axis="y", color=GRID, linewidth=1.0, zorder=0)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(INK)
    ax.spines[spine].set_linewidth(1.4)

# reference dashed lines
ax.axhline(BASELINE, color=GRAY, linewidth=2, linestyle=(0, (4, 6)), zorder=2)

# training curve
ax.plot(rounds, scores, color=BLUE, linewidth=3.5, zorder=3)
ax.scatter(rounds, scores, s=70, color=BLUE, edgecolors="white", linewidths=2, zorder=4)
for x, y in zip(rounds, scores):
    ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=10.5, color=INK, fontweight="bold")

ax.set_xlabel("Evaluation step", fontsize=12, color="#374151")

fig.tight_layout()
png_out = "docs/figures/meeting_policy_rl_swarm14b_validation_projected_20260529.png"
svg_out = "docs/figures/meeting_policy_rl_swarm14b_validation_projected_20260529.svg"
fig.savefig(png_out, dpi=140, facecolor="white")
fig.savefig(svg_out, facecolor="white")
print("wrote", png_out)
print("wrote", svg_out)
