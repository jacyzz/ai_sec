"""张量小工具：IoU、坐标转换等。"""
from __future__ import annotations

import torch


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """两集合 xyxy 框之间的 IoU 矩阵。

    boxes1: [N, 4], boxes2: [M, 4] -> [N, M]
    """
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-8)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def to_numpy(t: torch.Tensor):
    return t.detach().cpu().numpy()
