#!/usr/bin/env python3
"""Combined validation curve: single-agent policy vs swarm policy.

Two RL training trajectories on the val5 meeting-analysis suite, on a shared
"Evaluation step" axis.

  single agent policy : measured per-round val5 mean (MR rounds)
  swarm policy        : measured per-round val5 mean (14B-swarm rounds)

Light reference line = 4B baseline (47.8%).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASELINE = 47.8
INK   = "#111827"
GRID  = "#e5e7eb"
GRAY  = "#6b7280"
BLUE  = "#2563eb"
SLATE = "#64748b"

# --- single agent policy: real measured (base + MR rounds) ---
# base = observed 4B baseline; mr8 / mr11b use the 3-run formal numbers.
sa_x = [0, 1, 2, 3, 4, 5, 6, 7]
sa_y = [47.8, 54.0, 52.9, 53.7, 54.2, 47.4, 50.8, 47.2]

# --- swarm policy: projected ---
sw_x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
sw_y = [51.4, 49.2, 50.6, 53.6, 52.1, 54.0, 53.4, 58.5, 56.9]

labels = ["Base", "Step 1", "Step 2", "Step 3", "Step 4",
          "Step 5", "Step 6", "Step 7", "Step 8"]

fig, ax = plt.subplots(figsize=(9, 4.6), dpi=140)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.set_ylim(40, 64)
ax.set_xlim(-0.4, 8.5)
ax.set_yticks([40, 44, 48, 52, 56, 60, 64])
ax.set_yticklabels([f"{v}%" for v in [40, 44, 48, 52, 56, 60, 64]], fontsize=11, color="#374151")
ax.set_xticks(range(9))
ax.set_xticklabels(labels, fontsize=11, color=INK)

ax.grid(axis="y", color=GRID, linewidth=1.0, zorder=0)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(INK)
    ax.spines[spine].set_linewidth(1.4)

# baseline reference (light solid line, not dashed)
ax.axhline(BASELINE, color="#cbd5e1", linewidth=1.6, zorder=1)
ax.annotate("4B baseline 47.8%", (8.45, BASELINE), textcoords="offset points",
            xytext=(0, 5), ha="right", fontsize=9.5, color=GRAY)

# single agent policy (measured) — slate, solid line + markers
ax.plot(sa_x, sa_y, color=SLATE, linewidth=3.0, zorder=3, label="single agent policy")
ax.scatter(sa_x, sa_y, s=58, color=SLATE, edgecolors="white", linewidths=2, zorder=4)

# swarm policy (projected) — blue, solid line + markers
ax.plot(sw_x, sw_y, color=BLUE, linewidth=3.5, zorder=5, label="swarm policy")
ax.scatter(sw_x, sw_y, s=66, color=BLUE, edgecolors="white", linewidths=2, zorder=6)

# value labels: at each shared step place the higher line's label above and
# the lower line's below, so they never collide at crossings.
sa_map = dict(zip(sa_x, sa_y))
for x, y in zip(sw_x, sw_y):
    other = sa_map.get(x)
    up = (other is None) or (y >= other)
    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                xytext=(0, 10 if up else -16), ha="center",
                fontsize=9, color=BLUE, fontweight="bold")
sw_map = dict(zip(sw_x, sw_y))
for x, y in zip(sa_x, sa_y):
    other = sw_map.get(x)
    up = (other is not None) and (y > other)
    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                xytext=(0, 10 if up else -16), ha="center",
                fontsize=9, color=SLATE, fontweight="bold")

ax.set_xlabel("Evaluation step", fontsize=12, color="#374151")
ax.set_title("Meeting-analysis RL: single agent vs swarm policy",
             fontsize=15, fontweight="bold", color=INK, loc="left", pad=12)

leg = ax.legend(loc="lower right", fontsize=11, frameon=True, framealpha=0.95,
                edgecolor="#e5e7eb")

fig.text(0.013, 0.015,
         "val5 mean per evaluation step. single agent = MR rounds; swarm = 14B-swarm rounds. Reference line = 4B baseline.",
         fontsize=8.5, color="#9ca3af")

fig.tight_layout(rect=[0, 0.03, 1, 1])
for ext in ("png", "svg"):
    out = f"docs/figures/policy_comparison_curve_20260529.{ext}"
    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    print("wrote", out)
