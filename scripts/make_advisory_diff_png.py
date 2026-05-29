#!/usr/bin/env python3
"""Side-by-side content diff: same advisory_stakeholders task, Base 4B vs RL LoRA output.

Visualizes WHERE the RL improvement is at the content level (not just scores).
Both excerpts are verbatim from real bench transcripts:
  Base = run_20260529_qwen3_base_single_bench_test
  RL   = run_20260529_qwen_r08_lora_single_bench_test (round_08 LoRA)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle

for _fp in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/PingFang.ttc"):
    try:
        font_manager.fontManager.addfont(_fp)
    except Exception:
        pass
plt.rcParams["font.family"] = ["Arial Unicode MS", "PingFang SC", "DejaVu Sans"]

INK   = "#111827"
SUB   = "#6b7280"
MUTE  = "#9ca3af"
GRAYH = "#64748b"
BLUE  = "#2563eb"
BLUEH = "#1d4ed8"
GREEN = "#166534"
HL    = "#dcfce7"   # green highlight fill
HLED  = "#86efac"
FADE  = "#9ca3af"

fig, ax = plt.subplots(figsize=(13.6, 10.4), dpi=120)
fig.patch.set_facecolor("white")
W, H = 1360, 1040
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

# ---- title ----
ax.text(48, 44, "advisory_stakeholders：同一份会议记录，RL 前后产出对比",
        fontsize=23, fontweight="bold", color=INK, va="center")
ax.text(48, 76, "左右是同一任务的真实 deliverable 节选。RL 侧绿色高亮 = base 没写出来、RL 学会写的内容。",
        fontsize=13, color=SUB, va="center")

# ---- column headers ----
colL_x, colR_x = 48, 700
colW = 612
hdr_y = 108
ax.add_patch(FancyBboxPatch((colL_x, hdr_y), colW, 44, boxstyle="round,pad=0,rounding_size=8",
                            fc="#f1f5f9", ec="#e2e8f0", lw=1))
ax.text(colL_x+18, hdr_y+22, "BASE 4B", fontsize=15, fontweight="bold", color=GRAYH, va="center")
ax.text(colL_x+colW-18, hdr_y+22, "泛化 · 含糊 · 无后续", fontsize=11.5, color=MUTE, ha="right", va="center")

ax.add_patch(FancyBboxPatch((colR_x, hdr_y), colW, 44, boxstyle="round,pad=0,rounding_size=8",
                            fc="#eff6ff", ec="#bfdbfe", lw=1))
ax.text(colR_x+18, hdr_y+22, "RL POLICY (LoRA)", fontsize=15, fontweight="bold", color=BLUEH, va="center")
ax.text(colR_x+colW-18, hdr_y+22, "具名 · 量化 · 可执行", fontsize=11.5, color=BLUE, ha="right", va="center")

# ---- diff rows: (label, base_text, rl_text, rl_is_highlight) ----
rows = [
    ("利益相关者识别",
     "Commercial Wireless Carriers\n(泛指，无具体公司/人名)",
     "Verizon (Molly Feldman) · AT&T (Carl Povelites)\nShared Spectrum (Mark McHenry) · NOAA",
     True),
    ("具体成员立场",
     "(无任何点名发言者)",
     "Karl Nebbia (NTIA) · Tom Power (OSTP)\nDr. Charles Rush · Dr. Harold Furchtgott-Roth",
     True),
    ("迁移成本",
     "“High costs of relocation”\n(只有定性描述)",
     "“$18B for 1755-1850 band”\n(抓到了原文里的量化数字)",
     True),
    ("后续行动 / Next Steps",
     "(无 Next Steps 段落)",
     "Five working groups · co-chairs\ntechnical standards for interference mitigation",
     True),
    ("结构组织",
     "平铺 6 条列表",
     "按 Government / Commercial / Other\n分层组织",
     True),
    ("证据引用 (原文锚定)",
     "无逐句证据引用",
     "无逐句证据引用 — 两边都未解决",
     False),
]

y = 178
label_h = 26   # space reserved for the row label above the boxes
box_h = 78
row_gap = 22
for label, btxt, rtxt, hl in rows:
    # row label sits in its own band, clearly above both boxes
    ax.text(colL_x, y, label, fontsize=12.5, fontweight="bold", color=INK, va="center")
    by = y + label_h          # box top
    cy = by + box_h/2         # box vertical center
    # base box
    ax.add_patch(FancyBboxPatch((colL_x, by), colW, box_h, boxstyle="round,pad=0,rounding_size=8",
                                fc="#f8fafc", ec="#e5e7eb", lw=1))
    ax.text(colL_x+18, cy, btxt, fontsize=12.5, color=FADE, va="center")
    # rl box
    if hl:
        ax.add_patch(FancyBboxPatch((colR_x, by), colW, box_h, boxstyle="round,pad=0,rounding_size=8",
                                    fc=HL, ec=HLED, lw=1.4))
        ax.text(colR_x+18, cy, rtxt, fontsize=12.5, color="#14532d",
                fontweight="bold", va="center")
        ax.add_patch(FancyBboxPatch((colR_x+colW-58, by+12), 44, 22,
                                    boxstyle="round,pad=0,rounding_size=11", fc=GREEN, ec="none"))
        ax.text(colR_x+colW-36, by+23, "+", fontsize=16, fontweight="bold",
                color="white", ha="center", va="center")
    else:
        ax.add_patch(FancyBboxPatch((colR_x, by), colW, box_h, boxstyle="round,pad=0,rounding_size=8",
                                    fc="#f8fafc", ec="#e5e7eb", lw=1))
        ax.text(colR_x+18, cy, rtxt, fontsize=12.5, color=FADE, va="center")
    y = by + box_h + row_gap

# ---- footer ----
fy = y + 2
ax.text(48, fy, "提升集中在「把会议记录里的具体事实抽出来」：真名、真金额、可执行步骤。base 只会写正确但空泛的框架。",
        fontsize=12.5, color=INK, va="center")
ax.text(48, fy+24, "Source: val5 advisory_stakeholders 真实 deliverable 节选。Base = run_20260529_qwen3_base_single_bench_test；RL = round_08 LoRA。",
        fontsize=11, color=MUTE, va="center")

out = "docs/figures/advisory_stakeholders_diff_20260529.png"
fig.savefig(out, dpi=120, facecolor="white", bbox_inches="tight", pad_inches=0.22)
print("wrote", out)
