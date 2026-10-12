"""目标检测模型统一封装。

所有检测器都输出归一化后的 Detections（每张图一份 boxes/labels/scores），
攻击损失与评估都只依赖这个统一接口，因此可以任意插入自定义模型。
"""
from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import COCO_CLASSES


@dataclass
class Detections:
    """归一化检测结果：每个元素对应一张图。"""
    boxes: List[torch.Tensor] = field(default_factory=list)   # 每个 [N_i, 4] xyxy
    labels: List[torch.Tensor] = field(default_factory=list)  # 每个 [N_i] long
    scores: List[torch.Tensor] = field(default_factory=list)  # 每个 [N_i] float

    def __len__(self) -> int:
        return len(self.boxes)


class Detector(ABC):
    """把一批 [0,1] RGB 图像映射为 Detections。"""

    def __init__(self, device: torch.device):
        self.device = device

    @abstractmethod
    def forward(self, images: torch.Tensor) -> Detections:
        """images: [B, 3, H, W] float，范围 [0,1]。"""

    def __call__(self, images: torch.Tensor) -> Detections:
        return self.forward(images)


class TorchvisionDetector(Detector):
    """torchvision 的 Faster R-CNN / RetinaNet / FCOS（COCO 预训练）。"""

    _BUILDERS = {
        "fasterrcnn": "fasterrcnn_resnet50_fpn",
        "retinanet": "retinanet_resnet50_fpn",
        "fcos": "fcos_resnet50_fpn",
    }
    _WEIGHTS_ENUMS = {
        "fasterrcnn": "FasterRCNN_ResNet50_FPN_Weights",
        "retinanet": "RetinaNet_ResNet50_FPN_Weights",
        "fcos": "FCOS_ResNet50_FPN_Weights",
    }

    def __init__(self, name: str, device: torch.device):
        super().__init__(device)
        from torchvision.models import detection
        builder = getattr(detection, self._BUILDERS[name])
        self.model = builder(weights="DEFAULT").to(device).eval()
        self.name = name

        # torchvision 检测模型输出 1 起始的 91 类标签（person=1），而本框架统一用
        # COCO 80 类 0 起始（person=0）。用权重元信息把标签重映射到后者。
        weights_enum = getattr(detection, self._WEIGHTS_ENUMS[name])
        categories = weights_enum.DEFAULT.meta["categories"]
        label_map = torch.full((len(categories),), -1, dtype=torch.long)
        for lbl, cat in enumerate(categories):
            if cat in COCO_CLASSES:
                label_map[lbl] = COCO_CLASSES.index(cat)
        self._label_map = label_map.to(device)

    def forward(self, images: torch.Tensor) -> Detections:
        # 训练时 grad 开启，post-NMS 的 boxes/scores 仍保留梯度，可回传到补丁。
        out = self.model(list(images))
        return Detections(
            boxes=[o["boxes"] for o in out],
            labels=[self._label_map[o["labels"].long()] for o in out],
            scores=[o["scores"] for o in out],
        )


class ToyDetector(Detector):
    """冒烟测试用的小检测器：输出固定中心框，分数依赖图像内容（可微）。

    仅用于验证训练/评估流程能端到端跑通，不参与任何真实实验。
    """

    def __init__(self, device: torch.device, seed: int = 0):
        super().__init__(device)
        self.name = "toy"
        g = torch.Generator().manual_seed(seed)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        ).to(device)
        self.head = nn.Linear(16, 1).to(device)
        # 偏置设大一点，让干净图上的分数稳定高于阈值，便于观察攻击效果。
        nn.init.constant_(self.head.bias, 2.0)

    def forward(self, images) -> Detections:
        if isinstance(images, torch.Tensor):
            images = list(images)
        B = len(images)
        x = torch.stack(images, 0)  # 合成数据集图片尺寸一致，可 stack
        f = self.backbone(x).flatten(1)
        score = torch.sigmoid(self.head(f)).squeeze(1)  # [B]
        H, W = images[0].shape[-2], images[0].shape[-1]
        dev = images[0].device
        boxes = torch.tensor([0.15 * W, 0.15 * H, 0.85 * W, 0.85 * H], device=dev)
        boxes = boxes.unsqueeze(0).repeat(B, 1)
        labels = torch.zeros(B, dtype=torch.long, device=dev)
        return Detections(
            [boxes[i].unsqueeze(0) for i in range(B)],
            [labels[i].unsqueeze(0) for i in range(B)],
            [score[i].unsqueeze(0) for i in range(B)],
        )


class YoloDetector(Detector):
    """官方 ultralytics/yolov5 仓库封装（YOLOv5s 等锚点式检测器）。

    - 评估走标准 inference 路径：letterbox → 模型 → NMS → 缩放回原图坐标。
    - 训练走原始检测头（可微）：临时把 Detect 头置为 training，拿到原始
      `[bs, na, ny, nx, no]` 特征后手动解码，梯度可回传到补丁。
    """

    def __init__(
        self,
        weights: str = "yolov5s.pt",
        device: torch.device = None,
        target_idx: int = 0,
        repo_dir: Optional[str] = None,
        imgsz: int = 640,
        conf_thres: float = 0.05,
        iou_thres: float = 0.45,
    ):
        super().__init__(device)
        self.target_idx = target_idx
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.name = "yolov5s"

        repo = Path(repo_dir) if repo_dir else _default_yolov5_repo()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        # 权重路径解析到 yolov5 仓库内，避免相对路径随 cwd 变化而重复下载
        if weights and not os.path.isabs(weights) and os.path.dirname(weights) == "":
            weights = str(repo / weights)
        from models.experimental import attempt_load
        from utils.general import non_max_suppression

        self._nms = non_max_suppression
        self.model = attempt_load(weights, device=device)  # DetectionModel（fused + eval）
        self.detect = self.model.model[-1]  # Detect 头
        self.stride = int(self.model.stride.max().item())
        self.model.eval()

    def forward(self, images: torch.Tensor) -> Detections:
        if torch.is_grad_enabled():
            return self._forward_raw(images)
        return self._forward_predict(images)

    # ---- 预处理：letterbox（可微，torch 实现）----
    def _letterbox(self, img: torch.Tensor):
        """img: [3,H,W] float [0,1] RGB -> (padded [3,H',W'], scale r, left, top)。"""
        _, H, W = img.shape
        r = min(self.imgsz / H, self.imgsz / W)
        new_h = int(round(H * r))
        new_w = int(round(W * r))
        pad_h = (self.stride - new_h % self.stride) % self.stride
        pad_w = (self.stride - new_w % self.stride) % self.stride
        img = F.interpolate(img[None], size=(new_h, new_w), mode="bilinear", align_corners=False)[0]
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        img = F.pad(img, (left, right, top, bottom), value=114.0 / 255.0)
        return img, r, left, top

    @staticmethod
    def _scale_back(boxes: torch.Tensor, r: float, left: int, top: int) -> torch.Tensor:
        """把 letterbox 坐标缩放回原图坐标。boxes: [N,4] xyxy。"""
        if boxes.numel() == 0:
            return boxes
        shift = torch.tensor([left, top, left, top], device=boxes.device, dtype=boxes.dtype)
        return (boxes - shift) / r

    # ---- 训练路径：原始检测头，可微 ----
    def _forward_raw(self, images: torch.Tensor) -> Detections:
        boxes_all, labels_all, scores_all = [], [], []
        for img in images:
            x, r, left, top = self._letterbox(img)
            prev = self.detect.training
            self.detect.train()  # 让 Detect.forward 走训练分支，返回原始特征
            raw = self.model(x.unsqueeze(0))  # list of [1, na, ny, nx, no]
            self.detect.train(prev)

            boxes_parts, scores_parts = [], []
            for i, feat in enumerate(raw):
                _, _, ny, nx, _ = feat.shape
                grid, anchor_grid = self.detect._make_grid(nx, ny, i)
                y = feat.sigmoid()
                xy = (y[..., 0:2] * 2 + grid) * self.detect.stride[i]  # 中心点
                wh = (y[..., 2:4] * 2) ** 2 * anchor_grid          # 宽高
                obj = y[..., 4:5]                                   # objectness
                cls_person = y[..., 5 + self.target_idx:6 + self.target_idx]  # 目标类概率
                score = obj * cls_person                            # person 置信度
                cx, cy, w, h = xy[..., 0:1], xy[..., 1:2], wh[..., 0:1], wh[..., 1:2]
                b = torch.cat([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
                boxes_parts.append(b.reshape(-1, 4))
                scores_parts.append(score.reshape(-1))

            boxes = torch.cat(boxes_parts, 0).detach()  # 框只用于 IoU 匹配，无需梯度
            scores = torch.cat(scores_parts, 0)          # 分数保留梯度
            labels = torch.full_like(scores, self.target_idx, dtype=torch.long)
            boxes_all.append(self._scale_back(boxes, r, left, top))
            labels_all.append(labels)
            scores_all.append(scores)
        return Detections(boxes_all, labels_all, scores_all)

    # ---- 评估路径：NMS 推理，无梯度 ----
    def _forward_predict(self, images: torch.Tensor) -> Detections:
        boxes_all, labels_all, scores_all = [], [], []
        for img in images:
            x, r, left, top = self._letterbox(img)
            with torch.no_grad():
                pred = self.model(x.unsqueeze(0))[0]  # [1, 25200, 85]（xywh中心 + obj + cls）
                pred = self._nms(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres,
                                 classes=[self.target_idx])
            p = pred[0]  # [N,6] xyxy conf cls（可能为空）
            if p.numel() == 0:
                boxes = torch.zeros(0, 4, device=img.device)
                scores = torch.zeros(0, device=img.device)
                labels = torch.zeros(0, dtype=torch.long, device=img.device)
            else:
                boxes = self._scale_back(p[:, :4], r, left, top)
                scores = p[:, 4]
                labels = p[:, 5].long()
            boxes_all.append(boxes)
            labels_all.append(labels)
            scores_all.append(scores)
        return Detections(boxes_all, labels_all, scores_all)


def _default_yolov5_repo() -> Path:
    """默认 yolov5 仓库路径：<repo_root>/yolov5。"""
    here = Path(__file__).resolve().parent  # tupatch/
    repo_root = here.parent                   # ai_sec/
    return repo_root / "yolov5"


def build_detector(
    spec: str,
    device: torch.device,
    target_idx: int = 0,
    yolov5_repo: Optional[str] = None,
    yolo_imgsz: int = 640,
) -> Detector:
    """按名称或路径构造检测器。

    spec 支持: 'fasterrcnn' | 'retinanet' | 'fcos' | 'toy' | 'yolov5'/'yolov5s'，
    或以 .pt 结尾的路径 -> YoloDetector。
    """
    s = spec.lower()
    if s == "toy":
        return ToyDetector(device)
    if s in TorchvisionDetector._BUILDERS:
        return TorchvisionDetector(s, device)
    if s in ("yolov5", "yolov5s", "yolo"):
        return YoloDetector("yolov5s.pt", device, target_idx, repo_dir=yolov5_repo, imgsz=yolo_imgsz)
    if s.endswith(".pt"):
        return YoloDetector(spec, device, target_idx, repo_dir=yolov5_repo, imgsz=yolo_imgsz)
    raise ValueError(
        f"未知模型: {spec}。支持: fasterrcnn / retinanet / fcos / toy / yolov5 / yolov5s / *.pt"
    )
