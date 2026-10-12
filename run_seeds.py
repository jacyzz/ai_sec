"""Static vs Dynamic Ensemble 的多种子重复实验。

只改变随机种子，重复训练同一方法，用于报告 mean ± std，判断
「动态加权比静态平均好」这个结论是否超过种子间噪声。

数据划分（三个集合互不重叠）：
    train : train2017 含 person 图 [0:2000)
    val   : train2017 含 person 图 [2000:2500)   —— 开发期使用
    test  : val2017 含 person 图   [0:500)       —— final test，只在最后跑一次

用法：
    python run_seeds.py --coco_root data/coco --seeds 42 43 44 \
        --methods static dynamic --epochs 30 --out_dir outputs_seeds
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
from torch.utils.data import Subset

EVAL_MODELS = ["yolov5s", "fasterrcnn", "retinanet", "fcos"]
BLACKBOX = ["retinanet", "fcos"]

METHODS = {
    "static": {"dynamic_ensemble": False, "label": "Static Ensemble (equal weights)"},
    "dynamic": {"dynamic_ensemble": True, "label": "Dynamic Ensemble (ours)"},
}

TRAIN_N, VAL_START, VAL_N, TEST_N = 2000, 2000, 500, 500


def make_train_val(coco_root: str, cfg: Config, train_n: int = TRAIN_N, val_n: int = VAL_N):
    """train / val 共用一次 JSON 解析与预加载（只加载前 train_n+val_n 张，避免碰全量 64115）。"""
    tr_img = os.path.join(coco_root, "images", "train2017")
    ann_tr = os.path.join(coco_root, "annotations", "instances_train2017.json")
    ds_full = CocoPersonDataset(tr_img, ann_tr, cfg.target_idx,
                                train_n + val_n, cfg.max_side, start=0)
    ds_train = Subset(ds_full, range(0, train_n))
    ds_val = Subset(ds_full, range(train_n, train_n + val_n))
    return ds_train, ds_val


def make_test(coco_root: str, cfg: Config, test_n: int = TEST_N):
    te_img = os.path.join(coco_root, "images", "val2017")
    ann_te = os.path.join(coco_root, "annotations", "instances_val2017.json")
    return CocoPersonDataset(te_img, ann_te, cfg.target_idx, test_n, cfg.max_side, start=0)


def build_cfg(base: Config, method: str, seed: int, out_dir: str) -> Config:
    c = Config()
    for f in base.__dict__:
        setattr(c, f, getattr(base, f))
    c.surrogate_models = ["yolov5s", "fasterrcnn"]
    c.dynamic_ensemble = METHODS[method]["dynamic_ensemble"]
    c.use_momentum = False
    c.seed = seed
    c.out_dir = out_dir
    return c


def evaluate_all(cfg: Config, patch, ds, device):
    res = {}
    for spec in EVAL_MODELS:
        det = build_detector(spec, device, cfg.target_idx, cfg.yolov5_repo, cfg.yolo_imgsz)
        m = evaluate_detector(det, ds, patch, cfg, device)
        res[det.name] = {"asr": m["asr"], "clean_recall": m["clean_recall"]}
    res["_blackbox_avg"] = sum(res[m]["asr"] for m in BLACKBOX) / len(BLACKBOX)
    return res


def run_one(method: str, seed: int, base: Config, device: torch.device) -> dict:
    out_dir = os.path.join(base.out_dir, method, f"seed{seed}")
    cfg = build_cfg(base, method, seed, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ds_train, ds_val = make_train_val(base.coco_root, cfg)

    print(f"\n[{method} seed={seed}] 训练 dynamic_ensemble={cfg.dynamic_ensemble}", flush=True)
    t = time.time()
    surrogates = [build_detector(s, device, cfg.target_idx, cfg.yolov5_repo, cfg.yolo_imgsz)
                  for s in cfg.surrogate_models]
    patch = train_patch(cfg, surrogates, ds_train)
    p = patch.patch.detach()
    torch.save({"patch": p.cpu(), "config": cfg}, os.path.join(out_dir, "patch.pt"))
    cfg.save(os.path.join(out_dir, "config.json"))
    print(f"[{method} seed={seed}] 训练完成 {time.time()-t:.1f}s", flush=True)

    val = evaluate_all(cfg, p.to(device), ds_val, device)
    print(f"[{method} seed={seed}] val  黑盒平均 asr={val['_blackbox_avg']:.3f}", flush=True)
    # test 集只在最后构造，训练期间不占内存
    ds_test = make_test(base.coco_root, cfg)
    test = evaluate_all(cfg, p.to(device), ds_test, device)
    print(f"[{method} seed={seed}] test 黑盒平均 asr={test['_blackbox_avg']:.3f}", flush=True)

    out = {"val": val, "test": test}
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def summarize(summary: dict) -> None:
    """打印 mean ± std 表。"""
    for split in ("val", "test"):
        print(f"\n===== {split.upper()}  (mean ± std over seeds) =====", flush=True)
        print(f"{'method':>10s} " + " ".join(f"{m:>16s}" for m in EVAL_MODELS + ["BB_avg"]), flush=True)
        for method, per_seed in summary.items():
            runs = list(per_seed.values())
            cells = []
            for m in EVAL_MODELS:
                vals = [r[split][m]["asr"] for r in runs]
                cells.append(f"{_ms(vals):>16s}")
            bb = [r[split]["_blackbox_avg"] for r in runs]
            print(f"{method:>10s} " + " ".join(cells) + f"  {_ms(bb):>16s}", flush=True)
        print("  (格式: mean ± std, n=%d)" % len(runs), flush=True)


def _ms(vals) -> str:
    t = torch.tensor(vals, dtype=torch.float64)
    return f"{t.mean():.3f}±{t.std(unbiased=False):.3f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco_root", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--methods", nargs="+", default=["static", "dynamic"], choices=list(METHODS))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--out_dir", default="outputs_seeds")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    base = Config()
    base.coco_root = args.coco_root
    base.epochs = args.epochs
    base.batch_size = args.batch_size
    base.out_dir = args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)

    summary = {m: {} for m in args.methods}
    for method in args.methods:
        for seed in args.seeds:
            summary[method][str(seed)] = run_one(method, seed, base, device)
            with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

    summarize(summary)


if __name__ == "__main__":
    main()
