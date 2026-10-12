"""补丁参数化 + EOT 变换 + 贴图。

补丁 p ∈ [0,1]^{3×Hp×Wp}，训练时随机施加旋转/缩放/亮度/对比度变换（EOT），
再把变换后的补丁贴到目标 bbox 附近。贴图通过 grid_sample 实现，保证梯度可回传到补丁。

关于尺寸：Hp=Wp=64 是**可学习纹理的基础分辨率**，不是贴图的实际显示尺寸。
实际贴图边长由 `sqrt(patch_ratio * bbox面积)` 决定（见 `_transform_patch`），
64×64 纹理被重采样到该尺寸。因此同一份纹理可以直接应用到不同尺度的目标上，
无需为每个目标单独训练补丁。

注意这不等于尺度不变（scale invariance）——纹理被放大/缩小时攻击效果会变化，
补丁大小消融实验（run_patchsize.py）正是在量化这一依赖：黑盒平均 ASR 随
patch_ratio 从 0.280（3%）单调升至 0.800（20%）。
"""
from __future__ import annotations

import math
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Patch(nn.Module):
    """可学习正方形 RGB 补丁。"""

    def __init__(self, size: int, init: str = "uniform"):
        super().__init__()
        if init == "uniform":
            p = torch.rand(3, size, size)
        elif init == "zero":
            p = torch.zeros(3, size, size)
        else:
            raise ValueError(f"未知初始化方式: {init}")
        self.patch = nn.Parameter(p)

    def forward(self) -> torch.Tensor:
        return self.patch


def _build_theta(cx, cy, side, angle_deg, H, W, device, dtype):
    """构造 grid_sample 的仿射矩阵，把输出(图像)坐标映射回输入(补丁)坐标。

    side: 补丁在图像中占据的边长（像素）。angle_deg: 逆时针旋转角度。
    """
    c = math.cos(math.radians(angle_deg))
    s = math.sin(math.radians(angle_deg))
    theta = torch.zeros(2, 3, dtype=dtype, device=device)
    theta[0, 0] = (W / side) * c
    theta[0, 1] = (H / side) * s
    theta[0, 2] = (2.0 / side) * (c * (W / 2.0 - cx) + s * (H / 2.0 - cy))
    theta[1, 0] = -(W / side) * s
    theta[1, 1] = (H / side) * c
    theta[1, 2] = (2.0 / side) * (-s * (W / 2.0 - cx) + c * (H / 2.0 - cy))
    return theta


def _transform_patch(
    patch: torch.Tensor,
    box: torch.Tensor,
    H: int,
    W: int,
    cfg,
    rng: Optional[random.Random] = None,
    deterministic: bool = False,
):
    """对单个 bbox 施加一次变换，返回 (warped_patch [3,H,W], hard_mask [1,H,W])。"""
    rng = rng or random.Random(0)
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    side = math.sqrt(max(cfg.patch_ratio * bw * bh, 1.0))
    side = max(side, 4.0)

    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    if deterministic:
        angle, scale, bright, contrast = 0.0, 1.0, 0.0, 0.0
    else:
        angle = (rng.random() * 2.0 - 1.0) * cfg.eot_rotation_deg
        scale = rng.uniform(cfg.eot_scale_min, cfg.eot_scale_max)
        bright = (rng.random() * 2.0 - 1.0) * cfg.eot_brightness
        contrast = (rng.random() * 2.0 - 1.0) * cfg.eot_contrast
        if cfg.eot_position == "random":
            cx += (rng.random() - 0.5) * bw * 0.5
            cy += (rng.random() - 0.5) * bh * 0.5

    side *= scale

    # 亮度/对比度扰动（对补丁本身，可微）
    p = patch
    if bright != 0.0:
        p = (p * (1.0 + bright)).clamp(0.0, 1.0)
    if contrast != 0.0:
        p = ((p - 0.5) * (1.0 + contrast) + 0.5).clamp(0.0, 1.0)

    theta = _build_theta(cx, cy, side, angle, H, W, patch.device, patch.dtype)
    grid = F.affine_grid(theta.unsqueeze(0), size=(1, 3, H, W), align_corners=False)
    warped = F.grid_sample(
        p.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )  # [1,3,H,W]
    ones = torch.ones(1, 1, patch.shape[1], patch.shape[2], device=patch.device, dtype=patch.dtype)
    wmask = F.grid_sample(ones, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    hard_mask = (wmask > 0.5).float()  # [1,1,H,W]
    return warped[0], hard_mask[0]


def apply_patch(
    image: torch.Tensor,
    patch: torch.Tensor,
    boxes: torch.Tensor,
    cfg,
    rng: Optional[random.Random] = None,
    deterministic: bool = False,
) -> torch.Tensor:
    """把补丁贴到所有目标 bbox 上。

    image: [3, H, W] float 在 [0,1]。boxes: [M, 4] xyxy。返回对抗图 [3,H,W]。
    """
    adv = image.clone()
    H, W = image.shape[-2], image.shape[-1]
    for box in boxes:
        warped, mask = _transform_patch(patch, box, H, W, cfg, rng, deterministic)
        adv = adv * (1.0 - mask) + warped * mask
    return adv.clamp(0.0, 1.0)
