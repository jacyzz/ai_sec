"""生成报告素材：补丁图、干净 vs 攻击对比图、消融结果柱状图。"""
import os
import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from tupatch.config import Config
from tupatch.dataset import CocoPersonDataset
from tupatch.detectors import build_detector
from tupatch.patch import apply_patch
from tupatch.visualize import tensor_to_pil, save_patch


def load_patch(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["patch"].to(device), ckpt.get("config")


def draw_boxes(pil, boxes, color, width=3, labels=None):
    d = ImageDraw.Draw(pil)
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in b[:4]]
        d.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if labels is not None:
            d.text((x1, max(0, y1 - 12)), str(labels[i]), fill=color)
    return pil


def make_comparisons(method, patch, cfg, device, top_out_dir, n=5):
    """生成干净 vs 攻击对比图，红框=预测 person，绿框=GT。"""
    val_img = os.path.join(cfg.coco_root, "images", "val2017")
    ann_val = os.path.join(cfg.coco_root, "annotations", "instances_val2017.json")
    ds = CocoPersonDataset(val_img, ann_val, cfg.target_idx, 200, cfg.max_side)
    det = build_detector("yolov5s", device, cfg.target_idx, cfg.yolov5_repo, cfg.yolo_imgsz)
    out_dir = os.path.join(top_out_dir, method, "cmp")
    os.makedirs(out_dir, exist_ok=True)
    for k in range(n):
        image, boxes, img_id = ds[k]
        image = image.to(device); boxes = boxes.to(device)
        with torch.no_grad():
            clean = det(image.unsqueeze(0))
            adv = apply_patch(image, patch, boxes, cfg, deterministic=True)
            attack = det(adv.unsqueeze(0))
        # 过滤 person 预测（label==target_idx）
        def person_dets(detections):
            b = detections.boxes[0].cpu(); l = detections.labels[0].cpu(); s = detections.scores[0].cpu()
            keep = l == cfg.target_idx
            return b[keep], s[keep]
        cb, cs = person_dets(clean); ab, as_ = person_dets(attack)
        clean_pil = draw_boxes(tensor_to_pil(image), boxes.cpu(), "green", 3)
        clean_pil = draw_boxes(clean_pil, cb, "red", 2, labels=[f"{x:.2f}" for x in cs])
        adv_pil = draw_boxes(tensor_to_pil(adv), boxes.cpu(), "green", 3)
        adv_pil = draw_boxes(adv_pil, ab, "red", 2, labels=[f"{x:.2f}" for x in as_])
        w = clean_pil.width
        canvas = Image.new("RGB", (w * 2, clean_pil.height))
        canvas.paste(clean_pil, (0, 0)); canvas.paste(adv_pil, (w, 0))
        canvas.save(os.path.join(out_dir, f"cmp_{k:02d}_{img_id}.png"))
    print(f"[vis] {method} 对比图保存到 {out_dir}")


LABELS = {
    "baseline": "Baseline",
    "eot": "+EOT",
    "ensemble": "+Static Ensemble",
    "dynamic": "+Dynamic Ensemble\n(Ours)",
}


def make_chart(summary_path, out_path):
    """从 summary.json 画分组柱状图，并标注黑盒平均 ASR。"""
    with open(summary_path) as f:
        data = json.load(f)
    methods = [m for m in data.keys() if not m.startswith("_")]
    models = ["yolov5s", "fasterrcnn", "retinanet", "fcos"]
    blackbox = ["retinanet", "fcos"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(len(methods)))
    width = 0.18
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for j, m in enumerate(models):
        vals = []
        for meth in methods:
            r = data[meth].get(m)
            vals.append(r["asr"] if r else 0.0)
        ax.bar([xi + (j - 1.5) * width for xi in x], vals, width, label=m, color=colors[j])
    # 黑盒平均 ASR（用星号标注）
    for xi, meth in enumerate(methods):
        bb = [data[meth].get(m, {}).get("asr", 0.0) for m in blackbox]
        avg = sum(bb) / len(bb)
        ax.text(xi, 1.02, f"BB avg={avg:.3f}", ha="center", va="bottom", fontsize=8, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(m, m) for m in methods], fontsize=9)
    ax.set_ylabel("Attack Success Rate (ASR)")
    ax.legend(title="Detector")
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[chart] 柱状图保存到 {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--coco_root", required=True)
    p.add_argument("--out_dir", default="outputs_ablation")
    p.add_argument("--methods", nargs="+", default=["baseline", "eot", "ensemble", "dynamic"])
    p.add_argument("--n_cmp", type=int, default=5)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for m in args.methods:
        pp = os.path.join(args.out_dir, m, "patch.pt")
        if not os.path.exists(pp):
            print(f"跳过 {m}（无 patch.pt）")
            continue
        patch, cfg = load_patch(pp, device)
        if cfg is None:
            cfg = Config(); cfg.coco_root = args.coco_root; cfg.out_dir = args.out_dir
        else:
            cfg.coco_root = args.coco_root
        # 补丁图
        save_patch(patch, os.path.join(args.out_dir, m, "patch.png"))
        print(f"[vis] {m} 补丁图保存")
        # 对比图（只用 yolov5s 作为展示检测器）
        make_comparisons(m, patch, cfg, device, args.out_dir, args.n_cmp)

    summary = os.path.join(args.out_dir, "summary.json")
    if os.path.exists(summary):
        make_chart(summary, os.path.join(args.out_dir, "ablation_chart.png"))
