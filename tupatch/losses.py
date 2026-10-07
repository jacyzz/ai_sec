"""攻击损失与正则项。

主损失 L_conf：对每个 GT person，取与它 IoU 超过阈值的预测里置信度最高者 q_j，
目标是最小化这些 q_j 的均值，等价于让检测器"漏检"目标。
"""
from __future__ import annotations

from typing import List, Optional

import torch

from .utils import box_iou


def detection_suppress_loss(
    detections,
    target_boxes_list: List[torch.Tensor],
    target_idx: int,
    iou_thr: float = 0.3,
) -> Optional[torch.Tensor]:
    """Object-vanishing 损失。

    detections: Detections（每张图一份预测）。
    target_boxes_list: 每张图的 GT person 框 [M, 4]。
    返回所有命中目标的最大置信度均值；若没有可攻击目标则返回 None。
    """
    terms: List[torch.Tensor] = []
    for det_boxes, det_labels, det_scores, gt in zip(
        detections.boxes, detections.labels, detections.scores, target_boxes_list
    ):
        if gt.numel() == 0 or det_boxes.numel() == 0:
            continue
        iou = box_iou(gt, det_boxes)                 # [M, N]
        person = det_labels == target_idx            # [N]
        for m in range(gt.shape[0]):
            valid = person & (iou[m] > iou_thr)
            if valid.any():
                terms.append(det_scores[valid].max())
    if len(terms) == 0:
        return None
    return torch.stack(terms).mean()


def tv_loss(patch: torch.Tensor) -> torch.Tensor:
    """总变差损失，抑制补丁中的高频噪声。patch: [3, H, W]"""
    dh = (patch[:, 1:, :] - patch[:, :-1, :]).abs().mean()
    dw = (patch[:, :, 1:] - patch[:, :, :-1]).abs().mean()
    return dh + dw
