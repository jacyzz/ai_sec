"""命令行入口。

子命令：
    smoke   冒烟测试（无需 COCO / 真实模型，验证全流程）
    train   在替代模型上训练通用补丁
    eval    在多个检测器上评估补丁（白盒 + 黑盒）
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from .config import Config
from .dataset import CocoPersonDataset, SyntheticPersonDataset
from .detectors import build_detector
from .attack import train_patch
from .evaluate import evaluate_detector
from .visualize import save_examples, save_patch


def _coco_paths(cfg: Config):
    if cfg.coco_root is None:
        raise ValueError("需要提供 --coco_root（目录含 images/{train,val}2017 与 annotations/）")
    train_img = os.path.join(cfg.coco_root, "images", "train2017")
    val_img = os.path.join(cfg.coco_root, "images", "val2017")
    ann_train = cfg.ann_train or os.path.join(cfg.coco_root, "annotations", "instances_train2017.json")
    ann_val = cfg.ann_val or os.path.join(cfg.coco_root, "annotations", "instances_val2017.json")
    return train_img, val_img, ann_train, ann_val


def run_smoke(cfg: Config) -> None:
    device = cfg.resolve_device()
    print(f"[smoke] device={device}")

    # 冒烟测试固定用 toy 检测器，避免下载真实模型权重。
    surrogates = [build_detector("toy", device, cfg.target_idx)]
    ds = SyntheticPersonDataset(cfg.smoke_images, size=cfg.smoke_size, seed=cfg.seed)

    # 控制训练规模：跑 3 个 epoch、逐步更新，验证流程即可。
    cfg.epochs = 3
    cfg.batch_size = 1

    patch = train_patch(cfg, surrogates, ds)
    p = patch.patch.detach()

    os.makedirs(cfg.out_dir, exist_ok=True)
    save_patch(p, os.path.join(cfg.out_dir, "smoke_patch.png"))
    save_examples(ds, surrogates[0], p, cfg, device, os.path.join(cfg.out_dir, "smoke_examples"))

    metrics = evaluate_detector(surrogates[0], ds, p, cfg, device)
    print("[smoke] 结果:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}, ensure_ascii=False))


def run_train(cfg: Config) -> None:
    device = cfg.resolve_device()
    train_img, _, ann_train, _ = _coco_paths(cfg)
    ds = CocoPersonDataset(train_img, ann_train, cfg.target_idx, cfg.num_train, cfg.max_side)
    print(f"[train] 训练图片数={len(ds)}")

    surrogates = [
        build_detector(s, device, cfg.target_idx, cfg.yolov5_repo, cfg.yolo_imgsz)
        for s in cfg.surrogate_models
    ]
    print(f"[train] 替代模型={[s.name for s in surrogates]}")

    patch = train_patch(cfg, surrogates, ds)
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.save({"patch": patch.patch.detach().cpu(), "config": cfg}, os.path.join(cfg.out_dir, "patch.pt"))
    save_patch(patch.patch.detach(), os.path.join(cfg.out_dir, "patch.png"))
    cfg.save(os.path.join(cfg.out_dir, "config.json"))
    print(f"[train] 完成，补丁保存到 {cfg.out_dir}/patch.pt")


def run_eval(cfg: Config) -> None:
    device = cfg.resolve_device()
    _, val_img, _, ann_val = _coco_paths(cfg)
    ds = CocoPersonDataset(val_img, ann_val, cfg.target_idx, cfg.num_test, cfg.max_side)
    print(f"[eval] 测试图片数={len(ds)}")

    patch = None
    patch_path = os.path.join(cfg.out_dir, "patch.pt")
    if os.path.exists(patch_path):
        ckpt = torch.load(patch_path, map_location="cpu", weights_only=False)
        patch = ckpt["patch"]
        print("[eval] 已加载补丁")

    results = {}
    for spec in cfg.surrogate_models + cfg.blackbox_models:
        det = build_detector(spec, device, cfg.target_idx, cfg.yolov5_repo, cfg.yolo_imgsz)
        metrics = evaluate_detector(det, ds, patch, cfg, device)
        results[det.name] = metrics
        print(f"[eval] {det.name:16s} " + "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items() if v is not None
        ))

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main() -> None:
    p = argparse.ArgumentParser(description="TUPatch: Transferable Universal Patch for object detection")
    p.add_argument("command", choices=["smoke", "train", "eval"])
    p.add_argument("--config", type=str, default=None, help="JSON 配置文件，优先级低于命令行参数")
    p.add_argument("--coco_root", type=str, default=None)
    p.add_argument("--surrogates", nargs="+", default=None, help="替代模型列表")
    p.add_argument("--blackbox", nargs="+", default=None, help="黑盒模型列表（仅 eval 使用）")
    p.add_argument("--target", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None, help="梯度累积步数")
    p.add_argument("--num_eot", type=int, default=None, help="每张图每步 EOT 采样次数")
    p.add_argument("--patch_size", type=int, default=None)
    p.add_argument("--patch_ratio", type=float, default=None)
    p.add_argument("--use_momentum", type=lambda x: x.lower() in ("1", "true", "yes"), default=None)
    p.add_argument("--tv_lambda", type=float, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--yolov5_repo", type=str, default=None, help="yolov5 仓库路径（默认自动探测 <repo>/yolov5）")
    p.add_argument("--yolo_imgsz", type=int, default=None)
    p.add_argument("--num_train", type=int, default=None)
    p.add_argument("--num_test", type=int, default=None)
    p.add_argument("--max_side", type=int, default=None)
    args = p.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    # eval 默认复用训练时保存的 config.json，保证 surrogate/blackbox/target 与训练一致；
    # 命令行参数仍可覆盖。
    if args.command == "eval" and args.config is None:
        out_dir = args.out_dir or cfg.out_dir
        saved = os.path.join(out_dir, "config.json")
        if os.path.exists(saved):
            cfg = Config.load(saved)
    overrides = {
        "coco_root": args.coco_root,
        "surrogate_models": args.surrogates,
        "target_class": args.target,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "num_eot": args.num_eot,
        "patch_size": args.patch_size,
        "patch_ratio": args.patch_ratio,
        "use_momentum": args.use_momentum,
        "tv_lambda": args.tv_lambda,
        "out_dir": args.out_dir,
        "yolov5_repo": args.yolov5_repo,
        "yolo_imgsz": args.yolo_imgsz,
        "num_train": args.num_train,
        "num_test": args.num_test,
        "max_side": args.max_side,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    # blackbox 模型列表默认 RetinaNet，仅 eval 用
    cfg.blackbox_models = args.blackbox if args.blackbox is not None else ["retinanet"]

    if args.command == "smoke":
        run_smoke(cfg)
    elif args.command == "train":
        run_train(cfg)
    elif args.command == "eval":
        run_eval(cfg)


if __name__ == "__main__":
    main()
