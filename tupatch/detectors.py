"""目标检测模型统一封装。

所有检测器都输出归一化后的 Detections（每张图一份 boxes/labels/scores），
攻击损失与评估都只依赖这个统一接口，因此可以任意插入自定义模型。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn


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

    def __init__(self, name: str, device: torch.device):
        super().__init__(device)
        from torchvision.models import detection
        builder = getattr(detection, self._BUILDERS[name])
        self.model = builder(weights="DEFAULT").to(device).eval()
        self.name = name

    def forward(self, images: torch.Tensor) -> Detections:
        # 训练时 grad 开启，post-NMS 的 boxes/scores 仍保留梯度，可回传到补丁。
        out = self.model(list(images))
        return Detections(
            boxes=[o["boxes"] for o in out],
            labels=[o["labels"] for o in out],
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

    def forward(self, images: torch.Tensor) -> Detections:
        B = images.shape[0]
        f = self.backbone(images).flatten(1)
        score = torch.sigmoid(self.head(f)).squeeze(1)  # [B]
        H, W = images.shape[-2], images.shape[-1]
        boxes = torch.tensor([0.15 * W, 0.15 * H, 0.85 * W, 0.85 * H], device=images.device)
        boxes = boxes.unsqueeze(0).repeat(B, 1)
        labels = torch.zeros(B, dtype=torch.long, device=images.device)
        return Detections(
            [boxes[i].unsqueeze(0) for i in range(B)],
            [labels[i].unsqueeze(0) for i in range(B)],
            [score[i].unsqueeze(0) for i in range(B)],
        )


class YoloDetector(Detector):
    """Ultralytics YOLO 封装。

    - 评估走标准 predict 路径（结果已 detached，评估不需要梯度）。
    - 训练走原始检测头，使梯度能回传到补丁。原始路径按 YOLOv5 锚点式检测头编写，
      不同 ultralytics 版本的头结构有差异，使用时见 README 中的说明。
    """

    def __init__(self, model_path: str, device: torch.device, target_idx: int = 0):
        super().__init__(device)
        from ultralytics import YOLO
        self.yolo = YOLO(model_path)
        self.inner = self.yolo.model
        self.inner.eval().to(device)
        self.target_idx = target_idx
        self.name = "yolo"

    def forward(self, images: torch.Tensor) -> Detections:
        if torch.is_grad_enabled():
            return self._forward_raw(images)
        return self._forward_predict(images)

    def _forward_predict(self, images: torch.Tensor) -> Detections:
        res = self.yolo(images, verbose=False)
        return Detections(
            boxes=[r.boxes.xyxy for r in res],
            labels=[r.boxes.cls.long() for r in res],
            scores=[r.boxes.conf for r in res],
        )

    def _forward_raw(self, images: torch.Tensor) -> Detections:
        raise NotImplementedError(
            "YOLOv5 原始检测头（可微）解码需要在接入你的模型后按 ultralytics 版本适配，"
            "见 tupatch/README.md 中『YOLOv5 梯度路径』一节。评估（不训练）路径已可直接使用。"
        )


def build_detector(spec: str, device: torch.device, target_idx: int = 0) -> Detector:
    """按名称或路径构造检测器。

    spec 支持: 'fasterrcnn' | 'retinanet' | 'fcos' | 'toy'，
    或以 .pt 结尾 / 含 'yolo' 的路径 -> YoloDetector。
    """
    s = spec.lower()
    if s == "toy":
        return ToyDetector(device)
    if s in TorchvisionDetector._BUILDERS:
        return TorchvisionDetector(s, device)
    if s.endswith(".pt") or "yolo" in s:
        return YoloDetector(spec, device, target_idx)
    raise ValueError(
        f"未知模型: {spec}。支持: fasterrcnn / retinanet / fcos / toy / yolov5 权重路径"
    )
