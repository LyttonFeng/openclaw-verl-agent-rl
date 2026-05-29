#!/usr/bin/env python3
"""Render the advisory_stakeholders Base-vs-RL per-criterion improvement table as PNG.

All values are real grader breakdowns: base = run_20260529_qwen3_base_single_bench_test
(1-run), RL = MR8 round_08_temp07 (3-run mean).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

GRAY = "#94a3b8"
BLUE = "#2563eb"
INK = "#111827"
SUB = "#6b7280"
GREEN = "#166534"
MUTE = "#9ca3af"

# (criterion, base, rl, moved?)
rows = [
    ("Relocation cost captured",                 0.00, 0.67, True),
    ("Sharing preference stated",                0.00, 0.67, True),
    ("Stakeholder identification completeness",  0.25, 0.42, True),
    ("Conflict and agreement mapping",           0.25, 0.42, True),
    ("Interest and position analysis",           0.25, 0.38, True),
    ("Evidence-based analysis",                  0.00, 0.00, False),
    ("Specific member positions",                0.00, 0.00, False),
]

fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=120)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 1280)
ax.set_ylim(720, 0)
ax.axis("off")

ax.text(64, 50, "Advisory Stakeholders: Base 4B vs RL Policy",
        fontsize=24, fontweight="bold", color=INK, va="center")
ax.text(64, 86, "Per-criterion grader breakdown. Base = 1-run, RL (MR8) = 3-run mean. Real measured scores.",
        fontsize=12.5, color="#4b5563", va="center")

# overall banner
ax.add_patch(FancyBboxPatch((64, 120), 1152, 70, boxstyle="round,pad=0,rounding_size=8",
                            fc="#f8fafc", ec="#e5e7eb", lw=1))
ax.text(96, 155, "Overall score", fontsize=18, fontweight="bold", color=INK, va="center")
ax.text(520, 155, "40.8%", fontsize=20, fontweight="bold", color=SUB, ha="center", va="center")
ax.text(724, 155, "53.3%", fontsize=20, fontweight="bold", color=BLUE, ha="center", va="center")
ax.add_patch(FancyBboxPatch((880, 136), 246, 38, boxstyle="round,pad=0,rounding_size=19",
                            fc="#dcfce7", ec="none"))
ax.text(1003, 155, "+12.5 pp", fontsize=17, fontweight="bold", color=GREEN, ha="center", va="center")

# column headers
ax.text(96, 232, "CRITERION", fontsize=12, fontweight="bold", color=SUB, va="center")
ax.text(520, 232, "BASE 4B", fontsize=12, fontweight="bold", color=SUB, ha="center", va="center")
ax.text(724, 232, "RL POLICY", fontsize=12, fontweight="bold", color=SUB, ha="center", va="center")
ax.text(1003, 232, "GAIN", fontsize=12, fontweight="bold", color=SUB, ha="center", va="center")

BAR_W = 150.0
y0 = 268
rh = 50
for i, (name, base, rl, moved) in enumerate(rows):
    y = y0 + i * rh
    if i % 2 == 1:
        ax.add_patch(plt.Rectangle((64, y), 1152, rh, fc="#f8fafc", ec="#eef2f7", lw=1))
    else:
        ax.add_patch(plt.Rectangle((64, y), 1152, rh, fc="#ffffff", ec="#eef2f7", lw=1))
    cy = y + rh / 2
    ax.text(96, cy, name, fontsize=15, color=(INK if moved else SUB), va="center")
    by = cy - 9
    # base track + fill
    ax.add_patch(FancyBboxPatch((410, by), BAR_W, 18, boxstyle="round,pad=0,rounding_size=9",
                                fc="#e5e7eb", ec="none"))
    if base > 0:
        ax.add_patch(FancyBboxPatch((410, by), BAR_W * base, 18, boxstyle="round,pad=0,rounding_size=9",
                                    fc=GRAY, ec="none"))
    ax.text(580, cy, f"{int(round(base*100))}%", fontsize=13,
            color=(SUB if base > 0 else MUTE), va="center")
    # rl track + fill
    ax.add_patch(FancyBboxPatch((620, by), BAR_W, 18, boxstyle="round,pad=0,rounding_size=9",
                                fc=("#dbeafe" if moved else "#e5e7eb"), ec="none"))
    if rl > 0:
        ax.add_patch(FancyBboxPatch((620, by), BAR_W * rl, 18, boxstyle="round,pad=0,rounding_size=9",
                                    fc=BLUE, ec="none"))
    ax.text(790, cy, f"{int(round(rl*100))}%", fontsize=13,
            color=(BLUE if rl > 0 else MUTE), va="center")
    # gain
    if moved:
        ax.text(1003, cy, f"+{int(round((rl-base)*100))} pp", fontsize=14,
                fontweight="bold", color=GREEN, ha="center", va="center")
    else:
        ax.text(1003, cy, "—", fontsize=14, fontweight="bold", color=MUTE, ha="center", va="center")

ax.text(64, 642, "Source: Val5 advisory_stakeholders. Base = run_20260529_qwen3_base_single_bench_test. RL = MR8 (round_08_temp07), 3-run mean.",
        fontsize=12, color=SUB, va="center")
ax.text(64, 666, "RL gains concentrate on relocation cost & sharing-preference extraction; evidence-grounding and per-member positions remain unsolved by both.",
        fontsize=12, color=MUTE, va="center")

out = "docs/figures/advisory_stakeholders_improvement_table_20260528.png"
fig.savefig(out, dpi=120, facecolor="white", bbox_inches="tight", pad_inches=0.2)
print("wrote", out)
