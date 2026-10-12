"""补丁大小消融的 trade-off 图：ASR vs patch_ratio。

左panel = 白盒（参与训练的 surrogate），右panel = 黑盒（从未见过的检测器）。
底部附数据表（也是低对比度色板的替代可读通道）。

用法：python make_patchsize_chart.py --summary outputs_patchsize/summary.json
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 调色板（dataviz 默认实例，light mode；已通过 validate_palette.js 全部检查）
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

WHITEBOX = [("yolov5s", "#2a78d6"), ("fasterrcnn", "#eb6834")]
BLACKBOX = [("retinanet", "#1baf7a"), ("fcos", "#eda100")]

DISPLAY = {"yolov5s": "YOLOv5s", "fasterrcnn": "Faster R-CNN",
           "retinanet": "RetinaNet", "fcos": "FCOS"}


def load(summary_path):
    """按 float(ratio) 归一化键名（summary 里同时存在 "0.10" 和 "0.2"）。"""
    with open(summary_path) as f:
        raw = json.load(f)
    data = {float(k): v for k, v in raw.items()}
    ratios = sorted(data)
    return data, ratios


def xs(ratios):
    return [r * 100 for r in ratios]


def style(ax, title, show_ylabel):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_xlabel("patch area / person bbox area  (%)", color=INK_2, fontsize=9.5)
    if show_ylabel:
        ax.set_ylabel("Attack Success Rate (ASR)", color=INK_2, fontsize=9.5)
    ax.set_title(title, color=INK, fontsize=11.5, fontweight="bold", pad=8)


def plot_group(ax, data, ratios, group, random_data=None):
    x = xs(ratios)
    # 末点标签上下错开，避免黑盒两条线收敛时重叠
    dy = [9, -9]
    for i, (name, color) in enumerate(group):
        y = [data[r][name]["asr"] for r in ratios]
        ax.plot(x, y, color=color, linewidth=2.0, marker="o", markersize=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=DISPLAY[name], zorder=3)
        ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), textcoords="offset points",
                    xytext=(8, dy[i]), va="center", fontsize=9, color=INK_2)
        # 随机噪声对照（同色虚线，不单独进 legend）
        if random_data is not None:
            yr = [random_data[r][name]["asr"] for r in ratios]
            ax.plot(x, yr, color=color, linewidth=1.4, linestyle=(0, (4, 2.5)),
                    alpha=0.85, zorder=2)

    handles, labels = ax.get_legend_handles_labels()
    if random_data is not None:
        handles.append(Line2D([], [], color=MUTED, linewidth=1.4, linestyle=(0, (4, 2.5))))
        labels.append("random patch (control)")
    ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=9,
              labelcolor=INK_2, handlelength=2.2)

    ax.set_xlim(1.0, 26.5)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.0f}%" for v in x])


def add_table(ax_region, data, ratios, random_data=None):
    """数据表：方法名为第一列（不用 rowLabels，避免被裁切）。低对比色板下的可读通道。"""
    ax_region.axis("off")
    blackbox = [n for n, _ in BLACKBOX]
    col_labels = ["method"] + [f"{r*100:.0f}%" for r in ratios]
    cell_text, emphasis = [], []  # emphasis: 0 普通, 1 黑盒平均, 2 高亮(增益)
    for name, _ in WHITEBOX + BLACKBOX:
        tag = "(W)" if any(name == n for n, _ in WHITEBOX) else "(B)"
        row = [f"{DISPLAY[name]} {tag}"]
        row += [f"{data[r][name]['asr']:.3f}" for r in ratios]
        cell_text.append(row); emphasis.append(0)
    avg = [sum(data[r][n]["asr"] for n in blackbox) / len(blackbox) for r in ratios]
    cell_text.append(["Black-box avg"] + [f"{v:.3f}" for v in avg]); emphasis.append(1)

    if random_data is not None:
        ravg = [sum(random_data[r][n]["asr"] for n in blackbox) / len(blackbox) for r in ratios]
        cell_text.append(["Random patch (BB avg)"] + [f"{v:.3f}" for v in ravg])
        emphasis.append(0)
        cell_text.append(["Gain over random"]
                         + [f"+{a-r:.3f}" for a, r in zip(avg, ravg)])
        emphasis.append(2)

    tbl = ax_region.table(cellText=cell_text, colLabels=col_labels,
                          cellLoc="center", loc="center",
                          colWidths=[0.30, 0.175, 0.175, 0.175, 0.175])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.45)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_text_props(color=INK_2, fontweight="bold")
            cell.set_facecolor(SURFACE)
        else:
            kind = emphasis[r - 1]
            color = "#006300" if kind == 2 else INK  # 增益行用 success 文本色
            cell.set_text_props(color=color, fontweight="bold" if kind else "normal")
            cell.set_facecolor("#f2f1ec" if kind == 1 else SURFACE)
        if c == 0:
            cell.get_text().set_ha("left")
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="outputs_patchsize/summary.json")
    ap.add_argument("--random", default="outputs_random/summary.json",
                    help="随机噪声对照；不存在则只画优化补丁")
    ap.add_argument("--out", default="outputs_patchsize/patchsize_tradeoff.png")
    args = ap.parse_args()

    data, ratios = load(args.summary)
    random_data = None
    if args.random and os.path.exists(args.random):
        random_data, _ = load(args.random)
        print(f"[chart] 已并入随机对照 {args.random}")
    else:
        print("[chart] 未找到随机对照，仅画优化补丁")

    fig = plt.figure(figsize=(11, 7.3), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.3, 1.55],
                          hspace=0.28, wspace=0.14,
                          left=0.065, right=0.975, top=0.885, bottom=0.075)

    axA = fig.add_subplot(gs[0, 0])
    style(axA, "White-box surrogates (trained on)", show_ylabel=True)
    plot_group(axA, data, ratios, WHITEBOX, random_data)

    axB = fig.add_subplot(gs[0, 1], sharey=axA)
    style(axB, "Black-box detectors (never seen)", show_ylabel=False)
    plot_group(axB, data, ratios, BLACKBOX, random_data)
    axB.tick_params(labelleft=False)

    fig.suptitle("Patch size vs. transfer attack success (COCO person)",
                 color=INK, fontsize=13.5, fontweight="bold", x=0.065, ha="left", y=0.965)

    tab_ax = fig.add_subplot(gs[1, :])
    add_table(tab_ax, data, ratios, random_data)

    fig.text(0.065, 0.020,
             "solid = optimized patch, dashed = random-noise control (occlusion floor)   |   "
             "median patch side:  ~11 / 14 / 19 / 27 px",
             color=MUTED, fontsize=8.2, ha="left")

    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"[chart] {args.out}")


if __name__ == "__main__":
    main()
