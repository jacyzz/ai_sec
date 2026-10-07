"""补丁训练主循环：EOT + 集成模型 + 动量。"""
from __future__ import annotations

import random
from typing import Callable, List, Optional

import torch
from torch.utils.data import DataLoader

from .config import Config
from .dataset import collate_single
from .detectors import Detector
from .losses import detection_suppress_loss, tv_loss
from .patch import Patch, apply_patch


def _optimize_step(patch: Patch, opt, momentum: torch.Tensor, cfg: Config) -> None:
    """读取累积梯度，施加动量后更新补丁，并清零梯度。"""
    grad = patch.patch.grad
    if grad is None:
        return
    with torch.no_grad():
        if cfg.normalize_grad:
            grad = grad / (grad.abs().sum() + 1e-8)
        if cfg.use_momentum:
            momentum.mul_(cfg.momentum_decay).add_(grad)
            g = momentum
        else:
            g = grad
        g = g.detach()
        if opt is None:
            # sign 更新（MI-FGSM 形式）
            patch.patch.data -= cfg.lr * g.sign()
            patch.patch.data.clamp_(0.0, 1.0)
            patch.patch.grad = None
    if opt is not None:
        patch.patch.grad = g
        opt.step()
        with torch.no_grad():
            patch.patch.data.clamp_(0.0, 1.0)
        patch.patch.grad = None


def train_patch(
    cfg: Config,
    surrogates: List[Detector],
    dataset,
    logger: Callable[[str], None] = print,
) -> Patch:
    device = cfg.resolve_device()
    torch.manual_seed(cfg.seed)

    patch = Patch(cfg.patch_size, cfg.patch_init).to(device)
    opt = torch.optim.Adam([patch.patch], lr=cfg.lr) if cfg.optimizer == "adam" else None
    momentum = torch.zeros_like(patch.patch)
    rng = random.Random(cfg.seed)

    n_models = len(surrogates)
    weights = cfg.ensemble_weights if cfg.ensemble_weights is not None else [1.0 / n_models] * n_models
    if len(weights) != n_models:
        raise ValueError(f"ensemble_weights 长度 {len(weights)} 与替代模型数 {n_models} 不匹配")

    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_single)

    global_step = 0
    step_in_batch = 0
    running_loss = 0.0
    for epoch in range(cfg.epochs):
        for sample in loader:
            image, boxes, _ = sample
            image = image.to(device)
            boxes = boxes.to(device)

            total = None
            for _ in range(cfg.num_eot):
                adv = apply_patch(image, patch.patch, boxes, cfg, rng=rng)
                for sur, w in zip(surrogates, weights):
                    det = sur(adv.unsqueeze(0))
                    loss = detection_suppress_loss(det, [boxes], cfg.target_idx, cfg.attack_iou_thr)
                    if loss is None:
                        continue
                    total = (w * loss) if total is None else (total + w * loss)

            if total is None:
                continue
            total = total / cfg.num_eot + cfg.tv_lambda * tv_loss(patch.patch)
            running_loss += total.detach().item()

            total.backward()
            step_in_batch += 1
            if step_in_batch >= cfg.batch_size:
                _optimize_step(patch, opt, momentum, cfg)
                global_step += 1
                if global_step % 10 == 0:
                    logger(
                        f"[epoch {epoch:02d} | step {global_step:04d}] "
                        f"loss={running_loss / (cfg.batch_size * 10):.4f}"
                    )
                running_loss = 0.0
                step_in_batch = 0

    if step_in_batch > 0:
        _optimize_step(patch, opt, momentum, cfg)

    return patch
