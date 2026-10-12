"""补丁训练主循环：EOT + 集成模型 + 动量。"""
from __future__ import annotations

import random
from typing import Callable, List, Optional

import torch
from torch.utils.data import DataLoader

from .config import Config
from .dataset import collate_batch
from .detectors import Detector
from .losses import detection_suppress_loss, tv_loss
from .patch import Patch, apply_patch


def _dynamic_weights(losses, init_conf, tau: float, device) -> torch.Tensor:
    """动态集成权重：当前置信度相对初始越高（越难攻击）的模型权重越大。

    h_m = S_m(t) / S_m(0)，w = softmax(h / tau)。S_m(0) 未记录全时退化为均匀权重。
    """
    n = len(losses)
    if any(c is None for c in init_conf):
        return torch.full((n,), 1.0 / n, device=device)
    h = torch.zeros(n, device=device)
    for i, (l, c0) in enumerate(zip(losses, init_conf)):
        if l is not None:
            h[i] = l.detach() / (c0 + 1e-6)
    return torch.softmax(h / tau, dim=0)


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

    # 真批处理：cfg.batch_size 张图作为一个 batch 一起前向/反向，充分利用 GPU 并行。
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_batch)

    # 动态加权集成：记录每个模型的初始 person 置信度 S_m(0)
    init_conf = [None] * n_models

    global_step = 0
    running_loss = 0.0
    for epoch in range(cfg.epochs):
        for images, boxes_list, _ in loader:
            images = [img.to(device) for img in images]
            boxes_list = [b.to(device) for b in boxes_list]

            # 每个模型在 num_eot 次变换下的平均 loss（S_m）
            per_model = [None] * n_models
            for _ in range(cfg.num_eot):
                advs = [apply_patch(img, patch.patch, b, cfg, rng=rng) for img, b in zip(images, boxes_list)]
                for mi, sur in enumerate(surrogates):
                    det = sur(advs)
                    loss = detection_suppress_loss(det, boxes_list, cfg.target_idx, cfg.attack_iou_thr)
                    if loss is not None:
                        per_model[mi] = loss if per_model[mi] is None else per_model[mi] + loss
            per_model = [(l / cfg.num_eot) if l is not None else None for l in per_model]

            # 记录初始置信度（首次出现匹配目标时）
            for mi, l in enumerate(per_model):
                if l is not None and init_conf[mi] is None:
                    init_conf[mi] = l.detach().item()

            # 动态权重 vs 静态权重
            w = _dynamic_weights(per_model, init_conf, cfg.ensemble_tau, device) if cfg.dynamic_ensemble else weights

            total = None
            for mi, l in enumerate(per_model):
                if l is None:
                    continue
                total = (w[mi] * l) if total is None else (total + w[mi] * l)

            if total is None:
                continue
            total = total + cfg.tv_lambda * tv_loss(patch.patch)
            running_loss += total.detach().item()

            total.backward()
            _optimize_step(patch, opt, momentum, cfg)
            global_step += 1
            if global_step % 10 == 0:
                logger(
                    f"[epoch {epoch:02d} | step {global_step:04d}] "
                    f"loss={running_loss / 10:.4f}"
                )
            running_loss = 0.0

    return patch
