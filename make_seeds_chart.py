"""多种子重复实验图：Static vs Dynamic 的黑盒平均 ASR（3 seed）。

形式选择：点图（strip plot）而非带误差棒的柱状图。因为这里的结论是
「两者重叠、差异不显著」，只有把每个 seed 画出来、让重叠可见，才诚实地
传达了这一点；柱状图会隐藏种子分布形态。

用法：
    python make_seeds_chart.py --summary outputs_seeds/summary.json \
        --out report/figures/fig_seeds_dotplot.png
"""
from __future__ import annotations

import argparse
import json
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# 与 make_patchsize_chart.py 同一套调色板；两色相邻对已过 validate_palette.js
# （CVD ΔE 24.7 protan / 33.6 normal，contrast ≥ 3:1，全部 PASS）。
COLORS = {"static": "#2a78d6", "dynamic": "#eb6834"}
LABELS = {"static": "Static Ensemble (equal weights)", "dynamic": "Dynamic Ensemble (ours)"}
SPLIT = "test"

# 主消融（单 seed 42）的黑盒平均，用于展示「单 draw」落在哪里
ABLATION = {"static": 0.485, "dynamic": 0.503}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="outputs_seeds/summary.json")
    p.add_argument("--out", default="report/figures/fig_seeds_dotplot.png")
    args = p.parse_args()

    with open(args.summary, encoding="utf-8") as f:
        data = json.load(f)

    rows = {}
    for m in ("static", "dynamic"):
        seeds = sorted(data[m], key=int)
        vals = [data[m][s][SPLIT]["_blackbox_avg"] for s in seeds]
        rows[m] = {"seeds": seeds, "vals": vals,
                   "mean": st.mean(vals), "std": st.pstdev(vals)}

    fig, ax = plt.subplots(figsize=(10.4, 3.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    Y = {"static": 1.0, "dynamic": 0.0}   # static 在上
    JIT = [-0.11, 0.0, 0.11]              # 仅用于避免点重叠的纵向错位，非数据

    for m in ("static", "dynamic"):
        y, c = Y[m], COLORS[m]
        r = rows[m]
        # ±1 std 区间
        ax.barh(y, 2 * r["std"], left=r["mean"] - r["std"], height=0.30,
                color=c, alpha=0.16, zorder=1, linewidth=0)
        # 逐 seed 点
        for k, (s, v) in enumerate(zip(r["seeds"], r["vals"])):
            ax.plot([v], [y + JIT[k]], marker="o", ms=8.5, mfc=c, mec=SURFACE,
                    mew=1.6, zorder=4)
            ax.annotate(f"seed {s}", (v, y + JIT[k]), xytext=(0, -13),
                        textcoords="offset points", ha="center", fontsize=7.4,
                        color=MUTED, zorder=5)
        # 均值
        ax.plot([r["mean"]], [y], marker="D", ms=9, mfc=c, mec=SURFACE, mew=1.2,
                zorder=6)
        ax.annotate(f"mean {r['mean']:.3f}   ±1 std {r['std']:.3f}",
                    (r["mean"], y), xytext=(0, 20), textcoords="offset points",
                    ha="center", fontsize=8.8, color=INK_2, zorder=6)
        # 单 seed 消融值（次要编码：空心菱形 + 灰）
        ax.plot([ABLATION[m]], [y], marker="D", ms=6.5, mfc="none", mec=MUTED,
                mew=1.3, zorder=3)

    # 单 seed 消融的图例（画一次）
    ax.plot([], [], marker="D", ms=6.5, mfc="none", mec=MUTED, mew=1.3, ls="none",
            label="main ablation value (single seed 42)")

    ax.set_yticks([Y["static"], Y["dynamic"]])
    ax.set_yticklabels([LABELS["static"], LABELS["dynamic"]], fontsize=9.5, color=INK)
    ax.set_ylim(-0.85, 1.60)
    ax.set_xlim(0.465, 0.575)
    ax.set_xlabel("black-box average ASR  (RetinaNet + FCOS, never trained)",
                  color=INK_2, fontsize=9.5, loc="right")
    ax.set_title("Multi-seed repeat: Static vs Dynamic Ensemble",
                 color=INK, fontsize=12, fontweight="bold", pad=26, loc="left")
    ax.tick_params(axis="x", colors=INK_2, labelsize=9)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    d = rows["dynamic"]["mean"] - rows["static"]["mean"]
    ax.annotate(
        f"Δ (Dynamic − Static) = {d:+.3f}\n"
        f"|Δ| < Static std ({rows['static']['std']:.3f})\n"
        f"→ seed ranges overlap; no reproducible advantage",
        xy=(0.985, 0.015), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=9, color=INK, linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.55", fc="#ffffff", ec=GRID, lw=0.9))

    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.155), frameon=False,
              fontsize=8.2, labelcolor=MUTED, handletextpad=0.6)

    fig.text(0.008, 0.005,
             "test split = 500 val2017 images (final test) · n=3 seeds per method · "
             "x-axis zoomed to show overlap (not zero-based) · points offset vertically only to avoid overplotting",
             color=MUTED, fontsize=7.4, ha="left")

    fig.subplots_adjust(left=0.245, right=0.985, top=0.80, bottom=0.30)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"已保存 {args.out}")
    for m in ("static", "dynamic"):
        r = rows[m]
        print(f"  {m:8s} mean={r['mean']:.4f} std={r['std']:.4f} vals={['%.3f' % v for v in r['vals']]}")


if __name__ == "__main__":
    main()
