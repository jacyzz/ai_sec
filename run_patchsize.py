"""补丁大小消融：固定 dynamic 方法（EOT + 动态加权集成），只改变 patch_ratio。

patch_ratio 是补丁面积占 person bbox 面积的比例，边长 = sqrt(ratio * bbox面积)。
0.10 复用 outputs_ablation/dynamic 的已有结果，其余档位重新训练。

用法：
    python run_patchsize.py --coco_root /path/to/coco \
        --ratios 0.03 0.05 0.20 --epochs 30 --num_train 2000 --num_test 500 --out_dir outputs_patchsize
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import torch

from tupatch.config import Config
from tupatch.dataset import CocoPersonDataset
from tupatch.detectors import build_detector
from tupatch.attack import train_patch
from tupatch.evaluate import evaluate_detector

EVAL_MODELS = ["yolov5s", "fasterrcnn", "retinanet", "fcos"]
BLACKBOX = ["retinanet", "fcos"]


def build_cfg(base: Config, ratio: float, out_dir: str) -> Config:
    c = Config()
    for f in base.__dict__:
        setattr(c, f, getattr(base, f))
    c.surrogate_models = ["yolov5s", "fasterrcnn"]
    c.dynamic_ensemble = True
    c.use_momentum = False
    c.patch_ratio = ratio
    c.out_dir = out_dir
    return c


def train_ratio(ratio: float, cfg: Config, device: torch.device) -> dict:
    out_dir = os.path.join(cfg.out_dir, f"ratio_{ratio}")
    c = build_cfg(cfg, ratio, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    train_img = os.path.join(c.coco_root, "images", "train2017")
    ann_train = os.path.join(c.coco_root, "annotations", "instances_train2017.json")
    val_img = os.path.join(c.coco_root, "images", "val2017")
    ann_val = os.path.join(c.coco_root, "annotations", "instances_val2017.json")
    ds_train = CocoPersonDataset(train_img, ann_train, c.target_idx, c.num_train, c.max_side)
    ds_val = CocoPersonDataset(val_img, ann_val, c.target_idx, c.num_test, c.max_side)

    surrogates = [build_detector(s, device, c.target_idx, c.yolov5_repo, c.yolo_imgsz) for s in c.surrogate_models]
    print(f"\n[ratio={ratio}] 训练（dynamic ensemble, patch_ratio={ratio}）", flush=True)
    t = time.time()
    patch = train_patch(c, surrogates, ds_train)
    p = patch.patch.detach()
    torch.save({"patch": p.cpu(), "config": c}, os.path.join(out_dir, "patch.pt"))
    c.save(os.path.join(out_dir, "config.json"))
    print(f"[ratio={ratio}] 训练完成 {time.time()-t:.1f}s", flush=True)

    results = {}
    for spec in EVAL_MODELS:
        det = build_detector(spec, device, c.target_idx, c.yolov5_repo, c.yolo_imgsz)
        m = evaluate_detector(det, ds_val, p.to(device), c, device)
        results[det.name] = {"clean_recall": m["clean_recall"], "attack_recall": m["attack_recall"], "asr": m["asr"]}
        print(f"[ratio={ratio}] {det.name:12s} asr={m['asr']:.3f}", flush=True)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco_root", required=True)
    p.add_argument("--ratios", nargs="+", type=float, default=[0.03, 0.05, 0.20])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--num_train", type=int, default=2000)
    p.add_argument("--num_test", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--out_dir", default="outputs_patchsize")
    p.add_argument("--dynamic_dir", default="outputs_ablation/dynamic", help="ratio=0.10 复用此目录")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    cfg.coco_root = args.coco_root
    cfg.epochs = args.epochs
    cfg.num_train = args.num_train
    cfg.num_test = args.num_test
    cfg.batch_size = args.batch_size
    cfg.out_dir = args.out_dir

    summary = {}
    # 0.10 复用 dynamic 已有结果
    dyn_res = os.path.join(args.dynamic_dir, "results.json")
    if os.path.exists(dyn_res):
        os.makedirs(os.path.join(args.out_dir, "ratio_0.1"), exist_ok=True)
        shutil.copy(dyn_res, os.path.join(args.out_dir, "ratio_0.1", "results.json"))
        with open(dyn_res) as f:
            summary["0.10"] = json.load(f)
        print("[ratio=0.10] 复用 dynamic 已有结果", flush=True)

    for ratio in args.ratios:
        summary[str(ratio)] = train_ratio(ratio, cfg, device)

    # 汇总 + 打表
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== 补丁大小消融汇总 =====", flush=True)
    print(f"{'ratio':>8s} " + " ".join(f"{m:>12s}" for m in EVAL_MODELS + ["BB_avg"]), flush=True)
    for ratio in sorted(summary, key=float):
        r = summary[ratio]
        cells = [f"{r[m]['asr']:.3f}" if m in r else "-" for m in EVAL_MODELS]
        bb = sum(r[m]["asr"] for m in BLACKBOX if m in r) / len(BLACKBOX)
        print(f"{float(ratio)*100:6.1f}% " + " ".join(f"{c:>12s}" for c in cells) + f"  {bb:>12.3f}", flush=True)


if __name__ == "__main__":
    main()
