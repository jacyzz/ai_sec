"""随机噪声补丁对照：不训练，直接贴一块随机噪声，测各档位的 ASR。

作用：确定「贴一块同样大小的方块」本身造成的遮挡/扰动基线。
如果某档位优化补丁的 ASR ≈ 随机补丁的 ASR，说明该档位的攻击力来自遮挡而非对抗优化。

用法：
    python run_random_control.py --coco_root data/coco \
        --ratios 0.03 0.05 0.10 0.20 --num_test 500 --out_dir outputs_random
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from tupatch.config import Config
from tupatch.dataset import CocoPersonDataset
from tupatch.detectors import build_detector
from tupatch.evaluate import evaluate_detector

EVAL_MODELS = ["yolov5s", "fasterrcnn", "retinanet", "fcos"]
BLACKBOX = ["retinanet", "fcos"]
N_SEEDS = 3  # 多个随机种子取平均，降低单次抽样方差


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco_root", required=True)
    p.add_argument("--ratios", nargs="+", type=float, default=[0.03, 0.05, 0.10, 0.20])
    p.add_argument("--num_test", type=int, default=500)
    p.add_argument("--out_dir", default="outputs_random")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    base = Config()
    base.coco_root = args.coco_root
    base.num_test = args.num_test

    val_img = os.path.join(args.coco_root, "images", "val2017")
    ann_val = os.path.join(args.coco_root, "annotations", "instances_val2017.json")
    ds_val = CocoPersonDataset(val_img, ann_val, base.target_idx, base.num_test, base.max_side)

    dets = {m: build_detector(m, device, base.target_idx, base.yolov5_repo, base.yolo_imgsz)
            for m in EVAL_MODELS}

    summary = {}
    for ratio in args.ratios:
        c = Config()
        for f in base.__dict__:
            setattr(c, f, getattr(base, f))
        c.patch_ratio = ratio

        acc = {m: [] for m in EVAL_MODELS}
        for seed in range(N_SEEDS):
            g = torch.Generator(device="cpu").manual_seed(1000 + seed)
            noise = torch.rand(3, c.patch_size, c.patch_size, generator=g).to(device)
            for m in EVAL_MODELS:
                r = evaluate_detector(dets[m], ds_val, noise, c, device)
                acc[m].append(r["asr"])
            print(f"[ratio={ratio} seed={seed}] " +
                  " ".join(f"{m}={acc[m][-1]:.3f}" for m in EVAL_MODELS), flush=True)

        res = {m: {"asr": sum(acc[m]) / len(acc[m]),
                   "asr_std": float(torch.tensor(acc[m]).std().item())} for m in EVAL_MODELS}
        bb = sum(res[m]["asr"] for m in BLACKBOX) / len(BLACKBOX)
        res["_blackbox_avg"] = bb
        summary[str(ratio)] = res
        print(f"[ratio={ratio}] 随机补丁黑盒平均 ASR = {bb:.3f}", flush=True)

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== 随机噪声补丁对照 =====", flush=True)
    print(f"{'ratio':>8s} " + " ".join(f"{m:>12s}" for m in EVAL_MODELS + ["BB_avg"]), flush=True)
    for ratio in sorted(summary, key=float):
        r = summary[ratio]
        cells = [f"{r[m]['asr']:.3f}" for m in EVAL_MODELS]
        print(f"{float(ratio)*100:6.1f}% " + " ".join(f"{c:>12s}" for c in cells)
              + f"  {r['_blackbox_avg']:>12.3f}", flush=True)


if __name__ == "__main__":
    main()
