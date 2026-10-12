"""评估指标：mAP50、Recall、ASR。

ASR 分母只统计干净图上被正确检测的目标（IoU≥0.5 且 score≥conf_thr），
加入补丁后若该目标不再被正确检测即计为攻击成功。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from .config import Config
from .detectors import Detections
from .patch import apply_patch
from .utils import box_iou


def _match_detections(gts_all, dets_all, target_idx, iou_thr, score_thr):
    """贪心匹配，返回每个 GT 是否被命中，以及用于 AP 的 tp/fp。

    dets_all 每个元素是一张图的 Detections（batch=1），因此 d.boxes[0] 为该图的
    [N_i,4] 预测框。
    """
    matched: List[List[bool]] = [[False] * len(g) for g in gts_all]
    records = []  # (score, img_idx, det_idx)
    for i, d in enumerate(dets_all):
        boxes, labels, scores = d.boxes[0], d.labels[0], d.scores[0]  # [N_i, ...]
        for j in range(len(boxes)):
            if int(labels[j]) != target_idx:
                continue
            if float(scores[j]) < score_thr:
                continue
            records.append((float(scores[j]), i, j))
    records.sort(key=lambda r: -r[0])

    tp = [0] * len(records)
    fp = [0] * len(records)
    for k, (_, i, j) in enumerate(records):
        d = dets_all[i]
        g = gts_all[i]
        if len(g) == 0:
            fp[k] = 1
            continue
        ious = box_iou(g, d.boxes[0][j].unsqueeze(0))[:, 0]
        best, best_iou = -1, iou_thr
        for m in range(len(g)):
            if matched[i][m]:
                continue
            if float(ious[m]) > best_iou:
                best_iou = float(ious[m])
                best = m
        if best >= 0:
            matched[i][best] = True
            tp[k] = 1
        else:
            fp[k] = 1
    return matched, tp, fp


def _ap_from(tp, fp, n_gt) -> float:
    """11 点插值 AP。"""
    tp = torch.tensor(tp, dtype=torch.float32).cumsum(0)
    fp = torch.tensor(fp, dtype=torch.float32).cumsum(0)
    prec = tp / (tp + fp).clamp(min=1)
    rec = tp / max(n_gt, 1)
    ap = 0.0
    for t in torch.arange(0.0, 1.0001, 0.1):
        mask = rec >= t
        ap += (prec[mask].max().item() if mask.any() else 0.0) / 11.0
    return float(ap)


def evaluate_detector(
    detector,
    dataset,
    patch: Optional[torch.Tensor],
    cfg: Config,
    device: torch.device,
) -> Dict[str, Optional[float]]:
    """在数据集上评估单个检测器。patch 为 None 时只算干净指标。"""
    gts_all: List[torch.Tensor] = []
    clean_dets: List[Detections] = []
    attack_dets: List[Detections] = []

    for sample in dataset:
        image, boxes, _ = sample
        image = image.to(device)
        boxes = boxes.to(device)
        gts_all.append(boxes)
        with torch.no_grad():
            clean_dets.append(detector(image.unsqueeze(0)))
            if patch is not None:
                adv = apply_patch(image, patch, boxes, cfg, deterministic=True)
                attack_dets.append(detector(adv.unsqueeze(0)))
        # 每个样本同步一次，避免异步内核积压导致 CUDA 分配器死锁（WSL2 下实测会挂起）。
        if device.type == "cuda":
            torch.cuda.synchronize()

    n_gt = sum(len(g) for g in gts_all)
    clean_matched, clean_tp, clean_fp = _match_detections(
        gts_all, clean_dets, cfg.target_idx, cfg.eval_iou_thr, cfg.conf_threshold
    )
    clean_ap = _ap_from(clean_tp, clean_fp, n_gt)
    clean_recall = sum(sum(m) for m in clean_matched) / max(n_gt, 1)
    n_clean_det = sum(sum(m) for m in clean_matched)

    if patch is None:
        return {
            "clean_ap": clean_ap,
            "clean_recall": clean_recall,
            "n_gt": n_gt,
            "n_clean_det": n_clean_det,
            "attack_ap": None,
            "attack_recall": None,
            "asr": None,
        }

    attack_matched, attack_tp, attack_fp = _match_detections(
        gts_all, attack_dets, cfg.target_idx, cfg.eval_iou_thr, cfg.conf_threshold
    )
    attack_ap = _ap_from(attack_tp, attack_fp, n_gt)
    attack_recall = sum(sum(m) for m in attack_matched) / max(n_gt, 1)

    success = 0
    for i in range(len(gts_all)):
        for m in range(len(gts_all[i])):
            if clean_matched[i][m] and not attack_matched[i][m]:
                success += 1
    asr = success / max(n_clean_det, 1)

    return {
        "clean_ap": clean_ap,
        "clean_recall": clean_recall,
        "n_gt": n_gt,
        "n_clean_det": n_clean_det,
        "attack_ap": attack_ap,
        "attack_recall": attack_recall,
        "asr": asr,
    }
