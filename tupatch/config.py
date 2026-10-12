"""实验配置。所有超参数集中在这里，可通过 JSON 加载/保存，方便复现与写报告。"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import torch

# COCO 80 类官方顺序，person 下标为 0。
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def class_index(name: str) -> int:
    return COCO_CLASSES.index(name)


@dataclass
class Config:
    # ---- 通用 ----
    seed: int = 42
    device: str = "auto"                 # "auto" | "cuda" | "cpu"
    out_dir: str = "outputs_tupatch"
    target_class: str = "person"

    # ---- 数据 ----
    coco_root: Optional[str] = None      # 目录结构: <coco_root>/images/{split}2017/ + annotations/instances_{split}2017.json
    ann_train: Optional[str] = None      # 可覆盖训练集标注 json 路径
    ann_val: Optional[str] = None        # 可覆盖验证集标注 json 路径
    num_train: int = 2000
    num_val: int = 500
    num_test: int = 500
    max_side: int = 0                    # 0 表示保持原图尺寸，>0 表示把长边缩放到该值（加速）

    # ---- 补丁 ----
    patch_size: int = 64                 # 补丁为正方形，边长（像素）
    patch_ratio: float = 0.1             # 补丁面积 / 目标 bbox 面积
    patch_init: str = "uniform"          # "uniform" | "zero"

    # ---- 替代模型（白盒）----
    surrogate_models: List[str] = field(default_factory=lambda: ["fasterrcnn"])
    ensemble_weights: Optional[List[float]] = None   # None -> 平均权重
    # ---- 黑盒模型（仅评估，绝不参与训练）----
    blackbox_models: List[str] = field(default_factory=lambda: ["retinanet"])

    # ---- YOLOv5（官方 ultralytics/yolov5 仓库）----
    yolov5_repo: Optional[str] = None      # yolov5 仓库路径，None 自动探测 <repo_root>/yolov5
    yolo_imgsz: int = 640                  # YOLO 输入尺寸（letterbox 目标边长）

    # ---- 优化 ----
    optimizer: str = "adam"              # "adam" | "sign"（sign 即 MI-FGSM 形式）
    lr: float = 0.01
    epochs: int = 30
    batch_size: int = 4                  # 每步梯度累积的图片数
    use_momentum: bool = False           # 梯度动量（实验表明对本任务无正收益，默认关闭）
    momentum_decay: float = 0.9
    normalize_grad: bool = True          # 梯度 L1 归一化
    tv_lambda: float = 0.1               # TV loss 权重

    # ---- 动态加权集成（Dynamic Reweighted Ensemble）----
    dynamic_ensemble: bool = False       # 是否按"当前最难攻击的模型权重越大"动态加权
    ensemble_tau: float = 0.5            # softmax 温度，越大权重越平滑

    # ---- EOT（期望变换）----
    eot_rotation_deg: float = 20.0       # 旋转角度范围 (+-)
    eot_scale_min: float = 0.7           # 缩放范围
    eot_scale_max: float = 1.3
    eot_brightness: float = 0.3          # 亮度扰动 (+-)
    eot_contrast: float = 0.2            # 对比度扰动 (+-)
    eot_position: str = "random"         # 补丁在 bbox 内的位置 "random" | "center"
    num_eot: int = 1                     # 每张图每步采样的 EOT 次数

    # ---- 攻击损失 ----
    attack_iou_thr: float = 0.3          # L_conf 中判定"匹配目标"的 IoU 阈值

    # ---- 评估 ----
    conf_threshold: float = 0.5          # 检测置信度阈值
    eval_iou_thr: float = 0.5            # 评估匹配 IoU 阈值

    # ---- 冒烟测试 ----
    smoke_steps: int = 30
    smoke_images: int = 32
    smoke_size: int = 128

    @property
    def target_idx(self) -> int:
        return class_index(self.target_class)

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def save(self, path: str) -> None:
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(**d)
