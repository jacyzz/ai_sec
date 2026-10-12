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
        preload: bool = True,
        start: int = 0,
    ):
        self.images_dir = images_dir
        self.target_idx = target_idx
        self.max_side = max_side
        self.start = start
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

        # entries 顺序 = 标注文件中 image 首次出现的顺序（dict 保序），全程不排序，
        # 保证 [start : start+max_images] 这个切片在不同运行/机器上完全可复现。
        self.entries = []  # (image_id, [bbox xywh list])
        for img_id, anns in anns_by_image.items():
            self.entries.append((img_id, anns))
        end = None if max_images is None else start + max_images
        self.entries = self.entries[start:end]

        self.to_tensor = transforms.ToTensor()

        # 预加载：把图片解码成 tensor 常驻内存，消除训练循环里的 NFS 读盘 + PIL 解码。
        # 数据集在 NFS 上且 DataLoader 单进程 batch=1 时，这是主要瓶颈。
        self._cache = None
        if preload:
            print(f"[dataset] 预加载 {len(self.entries)} 张图片到内存...", flush=True)
            self._cache = [self._load_entry(i) for i in range(len(self.entries))]
            print("[dataset] 预加载完成", flush=True)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def _load_entry(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self._cache is not None:
            return self._cache[idx]
        return self._load_entry(idx)


class SyntheticPersonDataset(Dataset):
    """冒烟测试：随机图 + 固定中心 person 框，配合 ToyDetector 使用。"""

    def __init__(self, n: int, size: int = 128, seed: int = 0):
        self.n = n
        self.size = size
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        for i in range(self.n):
            yield self[i]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if idx < 0 or idx >= self.n:
            raise IndexError(idx)
        g = torch.Generator().manual_seed(self.seed + idx)
        img = torch.rand(3, self.size, self.size, generator=g)
        s = float(self.size)
        box = torch.tensor([[0.15 * s, 0.15 * s, 0.85 * s, 0.85 * s]])
        return img, box, idx


def collate_single(batch):
    """DataLoader(batch_size=1) 用，直接返回单个样本，避免变长 bbox 被 stack。"""
    return batch[0]


def collate_batch(batch):
    """DataLoader 批处理：返回 (images list, boxes list, ids list)。

    图片尺寸可变，因此不 stack，而是以 list 交给检测器（torchvision 原生吃 list，
    YOLO 内部 letterbox 到统一尺寸）。
    """
    images = [b[0] for b in batch]
    boxes_list = [b[1] for b in batch]
    ids = [b[2] for b in batch]
    return images, boxes_list, ids
