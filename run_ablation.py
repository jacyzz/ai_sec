"""TUPatch 消融实验：Baseline / +EOT / +Momentum / +Ensemble / TUPatch。

每组在指定的替代模型上训练，然后统一在 YOLOv5s / Faster R-CNN / RetinaNet / FCOS
四个模型上评估。评估视角：参与训练的模型是白盒，其余是黑盒（迁移 ASR）。

用法：
    python run_ablation.py --coco_root /path/to/coco \
        --epochs 30 --num_train 2000 --num_test 500 --out_dir outputs_ablation
    # 只训练某几个方法：
    python run_ablation.py ... --methods ensemble tupatch
    # 只评估已保存的 patch（不训练）：
    python run_ablation.py ... --eval-only --methods baseline eot momentum tupatch ensemble
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from tupatch.config import Config
from tupatch.dataset import CocoPersonDataset
from tupatch.detectors import build_detector
from tupatch.attack import train_patch
from tupatch.evaluate import evaluate_detector

# 评估模型：前两个可能作为 surrogate（白盒），retinanet/fcos 永远不参与训练（黑盒）。
EVAL_MODELS = ["yolov5s", "fasterrcnn", "retinanet", "fcos"]


def make_config(coco_root, epochs, num_train, num_test, batch_size, out_dir):
    cfg = Config()
    cfg.coco_root = coco_root
    cfg.epochs = epochs
    cfg.num_train = num_train
    cfg.num_test = num_test
    cfg.batch_size = batch_size
    cfg.out_dir = out_dir
    return cfg


def no_eot(cfg: Config) -> Config:
    """关闭 EOT：确定性贴图（无旋转/缩放/亮度/位置扰动，单次采样）。"""
    cfg.eot_rotation_deg = 0.0
    cfg.eot_scale_min = 1.0
    cfg.eot_scale_max = 1.0
    cfg.eot_brightness = 0.0
    cfg.eot_contrast = 0.0
    cfg.eot_position = "center"
    cfg.num_eot = 1
    return cfg


# 四个方法，回答三个问题：
#   baseline -> +EOT          : Q1 EOT 有用吗？
#   +EOT     -> ensemble      : Q2 多模型训练能提升迁移吗？（静态平均）
#   ensemble -> dynamic       : Q3 动态权重比静态平均好吗？
ABLATIONS = {
    "baseline": {"surrogates": ["yolov5s"], "dynamic_ensemble": False, "post": no_eot},
    "eot": {"surrogates": ["yolov5s"], "dynamic_ensemble": False, "post": None},
    "ensemble": {"surrogates": ["yolov5s", "fasterrcnn"], "dynamic_ensemble": False, "post": None},
    "dynamic": {"surrogates": ["yolov5s", "fasterrcnn"], "dynamic_ensemble": True, "post": None},
}


def _build_cfg(spec: dict, base: Config, out_dir: str) -> Config:
    c = Config()
    for field in base.__dict__:
        setattr(c, field, getattr(base, field))
    c.surrogate_models = spec["surrogates"]
    c.use_momentum = False  # 实验表明动量对本任务无正收益，统一关闭
    c.dynamic_ensemble = spec["dynamic_ensemble"]
    c.out_dir = out_dir
    if spec["post"] is not None:
        spec["post"](c)
    return c


def _datasets(cfg: Config):
    train_img = os.path.join(cfg.coco_root, "images", "train2017")
    val_img = os.path.join(cfg.coco_root, "images", "val2017")
    ann_train = os.path.join(cfg.coco_root, "annotations", "instances_train2017.json")
    ann_val = os.path.join(cfg.coco_root, "annotations", "instances_val2017.json")
    ds_train = CocoPersonDataset(train_img, ann_train, cfg.target_idx, cfg.num_train, cfg.max_side)
    ds_val = CocoPersonDataset(val_img, ann_val, cfg.target_idx, cfg.num_test, cfg.max_side)
    return ds_train, ds_val


def train_patch_only(name: str, spec: dict, cfg: Config, device: torch.device) -> Config:
    c = _build_cfg(spec, cfg, os.path.join(cfg.out_dir, name))
    os.makedirs(c.out_dir, exist_ok=True)
    ds_train, _ = _datasets(c)
    surrogates = [build_detector(s, device, c.target_idx, c.yolov5_repo, c.yolo_imgsz) for s in c.surrogate_models]
    print(f"\n[{name}] 训练替代模型={[s.name for s in surrogates]}", flush=True)
    t = time.time()
    patch = train_patch(c, surrogates, ds_train)
    torch.save({"patch": patch.patch.detach().cpu(), "config": c}, os.path.join(c.out_dir, "patch.pt"))
    c.save(os.path.join(c.out_dir, "config.json"))
    print(f"[{name}] 训练完成 {time.time()-t:.1f}s", flush=True)
    return c


def eval_patch_only(name: str, cfg: Config, device: torch.device) -> dict:
    ckpt = torch.load(os.path.join(cfg.out_dir, "patch.pt"), map_location="cpu", weights_only=False)
    patch = ckpt["patch"]
    cfg = ckpt.get("config", cfg)
    _, ds_val = _datasets(cfg)

    results = {}
    for spec_name in EVAL_MODELS:
        det = build_detector(spec_name, device, cfg.target_idx, cfg.yolov5_repo, cfg.yolo_imgsz)
        m = evaluate_detector(det, ds_val, patch.to(device), cfg, device)
        results[det.name] = {
            "clean_recall": m["clean_recall"],
            "attack_recall": m["attack_recall"],
            "asr": m["asr"],
            "n_clean_det": m["n_clean_det"],
        }
        print(f"[{name}] {det.name:12s} clean_recall={m['clean_recall']:.3f} "
              f"attack_recall={m['attack_recall']:.3f} asr={m['asr']:.3f}", flush=True)

    results["_surrogates"] = cfg.surrogate_models
    with open(os.path.join(cfg.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def run_one(name: str, spec: dict, cfg: Config, device: torch.device, eval_only: bool) -> dict:
    c = _build_cfg(spec, cfg, os.path.join(cfg.out_dir, name))
    os.makedirs(c.out_dir, exist_ok=True)
    if not eval_only:
        train_patch_only(name, spec, cfg, device)
    return eval_patch_only(name, c, device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco_root", type=str, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--num_train", type=int, default=2000)
    p.add_argument("--num_test", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--out_dir", type=str, default="outputs_ablation")
    p.add_argument("--methods", nargs="+", default=None, help="默认全部；可指定子集")
    p.add_argument("--eval-only", action="store_true", help="只评估已保存的 patch，不训练")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    cfg = make_config(args.coco_root, args.epochs, args.num_train, args.num_test, args.batch_size, args.out_dir)

    methods = args.methods or list(ABLATIONS.keys())
    all_results = {}
    for name in methods:
        if name not in ABLATIONS:
            print(f"未知方法 {name}，跳过", flush=True)
            continue
        all_results[name] = run_one(name, ABLATIONS[name], cfg, device, args.eval_only)

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 打印一张带白盒/黑盒标注的汇总表
    print("\n===== 汇总（白盒=参与训练，黑盒=迁移）=====", flush=True)
    header = f"{'method':10s} " + " ".join(f"{m:>20s}" for m in EVAL_MODELS)
    print(header, flush=True)
    for name, res in all_results.items():
        surs = set(res.get("_surrogates", []))
        cells = []
        for m in EVAL_MODELS:
            r = res.get(m)
            if r is None:
                cells.append(f"{'-':>20s}")
                continue
            tag = "W" if m in surs else "B"
            cells.append(f"{r['asr']:.3f}({tag})".rjust(20))
        print(f"{name:10s} " + " ".join(cells), flush=True)
    print("(W=白盒 surrogate, B=黑盒迁移)", flush=True)


if __name__ == "__main__":
    main()
