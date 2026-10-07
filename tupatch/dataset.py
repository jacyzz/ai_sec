"""COCO person 数据集 + 冒烟测试用的合成数据集。

COCO 解析用手写 JSON（不依赖 pycocotools），只保留含 person 的图片。
"""
from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import class_index
from .utils import xywh_to_xyxy

_COCO_PERSON_ID = 1  # COCO 中 person 的 category id


class CocoPersonDataset(Dataset):
    """加载 COCO 标注，筛选含 person 的图片，返回 (image, person_boxes, image_id)。"""

    def __init__(
        self,
        images_dir: str,
        ann_file: str,
        target_idx: int = 0,
        max_images: Optional[int] = None,
        max_side: int = 0,
    ):
        self.images_dir = images_dir
        self.target_idx = target_idx
        self.max_side = max_side
        with open(ann_file, encoding="utf-8") as f:
            data = json.load(f)

        self._img_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
        self._img_id_to_size = {
            img["id"]: (img["width"], img["height"]) for img in data["images"]
        }
        # person 的 category_id 映射（COCO person = 1）
        self._person_cat_id = None
        for cat in data["categories"]:
            if cat["name"] == "person":
                self._person_cat_id = cat["id"]
                break
        if self._person_cat_id is None:
            self._person_cat_id = _COCO_PERSON_ID

        anns_by_image = {}
        for ann in data["annotations"]:
            if ann["category_id"] != self._person_cat_id:
                continue
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self.entries = []  # (image_id, [bbox xywh list])
        for img_id, anns in anns_by_image.items():
            self.entries.append((img_id, anns))
        if max_images is not None and len(self.entries) > max_images:
            self.entries = self.entries[:max_images]

        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img_id, anns = self.entries[idx]
        path = os.path.join(self.images_dir, self._img_id_to_file[img_id])
        img = Image.open(path).convert("RGB")
        orig_w, orig_h = self._img_id_to_size[img_id]

        boxes = torch.stack(
            [xywh_to_xyxy(torch.tensor(a["bbox"], dtype=torch.float32)) for a in anns]
        )  # [M, 4] xyxy in 原始像素坐标

        if self.max_side and max(orig_w, orig_h) > self.max_side:
            scale = self.max_side / float(max(orig_w, orig_h))
            new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
            img = img.resize((new_w, new_h), Image.BILINEAR)
            boxes = boxes * scale

        image = self.to_tensor(img)  # [3, H, W] float [0,1]
        return image, boxes, img_id


class SyntheticPersonDataset(Dataset):
    """冒烟测试：随机图 + 固定中心 person 框，配合 ToyDetector 使用。"""

    def __init__(self, n: int, size: int = 128, seed: int = 0):
        self.n = n
        self.size = size
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        g = torch.Generator().manual_seed(self.seed + idx)
        img = torch.rand(3, self.size, self.size, generator=g)
        s = float(self.size)
        box = torch.tensor([[0.15 * s, 0.15 * s, 0.85 * s, 0.85 * s]])
        return img, box, idx


def collate_single(batch):
    """DataLoader(batch_size=1) 用，直接返回单个样本，避免变长 bbox 被 stack。"""
    return batch[0]
