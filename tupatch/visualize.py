"""可视化：保存补丁图、干净/攻击对比图。"""
from __future__ import annotations

import os
from typing import List, Optional

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T

from .patch import apply_patch


def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    return T.ToPILImage()(img.clamp(0, 1).cpu())


def _draw_boxes(pil: Image.Image, boxes, labels=None, scores=None, color="red", width=3):
    draw = ImageDraw.Draw(pil)
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in b.tolist()]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        text = ""
        if labels is not None:
            text = f"{int(labels[i])}"
        if scores is not None:
            text += f" {float(scores[i]):.2f}"
        if text:
            draw.text((x1, max(0, y1 - 10)), text, fill=color)
    return pil


def save_patch(patch: torch.Tensor, path: str) -> None:
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    tensor_to_pil(patch.clamp(0, 1).cpu()).save(path)


def save_examples(
    dataset,
    detector,
    patch: Optional[torch.Tensor],
    cfg,
    device: torch.device,
    out_dir: str,
    n: int = 4,
) -> List[str]:
    """保存前 n 张干净 vs 攻击对比图（绿框=GT，红框=预测 person）。"""
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for k, sample in enumerate(dataset):
        if k >= n:
            break
        image, boxes, img_id = sample
        image = image.to(device)
        boxes = boxes.to(device)
        with torch.no_grad():
            clean = detector(image.unsqueeze(0))
            adv = apply_patch(image, patch, boxes, cfg, deterministic=True) if patch is not None else image
            attack = detector(adv.unsqueeze(0))
        if device.type == "cuda":
            torch.cuda.synchronize()

        clean_pil = tensor_to_pil(image)
        _draw_boxes(clean_pil, boxes, color="green", width=2)
        _draw_boxes(clean_pil, clean.boxes[0], clean.labels[0], clean.scores[0], color="red")

        adv_pil = tensor_to_pil(adv)
        _draw_boxes(adv_pil, boxes, color="green", width=2)
        _draw_boxes(adv_pil, attack.boxes[0], attack.labels[0], attack.scores[0], color="red")

        w = clean_pil.width
        canvas = Image.new("RGB", (w * 2, clean_pil.height))
        canvas.paste(clean_pil, (0, 0))
        canvas.paste(adv_pil, (w, 0))
        path = os.path.join(out_dir, f"cmp_{k:02d}_{img_id}.png")
        canvas.save(path)
        saved.append(path)
    return saved
