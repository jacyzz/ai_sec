"""TUPatch: Transferable Universal adversarial Patch for object detection.

实现「通用补丁 + EOT + 集成模型 + 动量」的可迁移通用对抗补丁攻击框架，
主任务为 Object Vanishing / Detection Suppression（让 person 目标漏检）。

用法示例见 README.md，或直接运行冒烟测试：
    python -m tupatch.main smoke
"""

import torch

# 本实验重计算都在 GPU 上，CPU 只做数据加载/小张量操作。torch 2.9.x 在部分
# WSL2 环境下多线程并行后端会在连续推理循环里死锁（基础算子 hang、CPU 100% 空转）。
# 关闭 CPU 多线程即可稳定规避，且对 GPU 训练/推理无影响。
torch.set_num_threads(1)

from .config import COCO_CLASSES, Config, class_index
from .detectors import Detections, Detector, build_detector
from .patch import Patch, apply_patch
from .attack import train_patch

__all__ = [
    "COCO_CLASSES",
    "Config",
    "class_index",
    "Detections",
    "Detector",
    "build_detector",
    "Patch",
    "apply_patch",
    "train_patch",
]

__version__ = "0.1.0"
