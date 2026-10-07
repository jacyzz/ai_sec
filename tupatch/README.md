# TUPatch —— 面向目标检测模型的可迁移通用对抗补丁

把「通用补丁 + EOT + 集成模型 + 动量」合成一条链路，主任务为 **Object Vanishing**：
干净图上能检测到的 `person`，加入补丁后应漏检。

```
COCO(person) → 替代模型(YOLOv5s + Faster R-CNN) → 通用补丁 → 黑盒模型(RetinaNet) → 迁移 ASR
```

## 目录结构

```
tupatch/
  config.py      所有超参数（可 JSON 存取）
  detectors.py   检测器统一接口 + torchvision / YOLO / toy 封装
  dataset.py     COCO person 数据集（手写 JSON 解析，不依赖 pycocotools）+ 合成数据集
  patch.py       补丁参数化 + EOT 变换 + grid_sample 贴图
  losses.py      L_conf（漏检损失）+ TV 正则
  attack.py      训练循环（EOT + 集成 + 动量）
  evaluate.py    mAP50 / Recall / ASR
  visualize.py   补丁图与对比图
  main.py        smoke / train / eval 三个入口
```

## 快速验证（无需 COCO 与真实模型）

```bash
conda activate llm
cd /mnt/d/project/hust/ai_sec
python -m tupatch.main smoke
```

这会用 `ToyDetector` + 合成数据集把「训练→贴图→评估→保存图」整条流程跑一遍，
确认没有 bug 后再接入真实模型/数据。

## 训练（你提供 COCO 后）

COCO 目录结构约定：

```
<coco_root>/
  images/train2017/  *.jpg
  images/val2017/    *.jpg
  annotations/instances_train2017.json
  annotations/instances_val2017.json
```

```bash
python -m tupatch.main train \
  --coco_root /path/to/coco \
  --surrogates fasterrcnn retinanet \
  --target person --epochs 30 --patch_size 64 --patch_ratio 0.1 \
  --out_dir outputs_tupatch
```

- `--surrogates` 可写 `fasterrcnn` / `retinanet` / `fcos`（torchvision 自动下载 COCO 权重），
  或 `.pt` 路径（走 YOLO 封装）。
- 集成：`--surrogates fasterrcnn retinanet` 即同时对两个模型取平均损失。

## 评估

```bash
python -m tupatch.main eval \
  --coco_root /path/to/coco \
  --surrogates fasterrcnn retinanet \
  --blackbox retinanet fcos \
  --out_dir outputs_tupatch
```

白盒/黑盒只是评估视角：参与训练的模型是白盒，`--blackbox` 里的模型**从未参与训练**，
其 ASR 就是迁移攻击成功率。评估会在 `outputs_tupatch/results.json` 输出各模型的
`clean_ap / clean_recall / attack_ap / attack_recall / asr`。

## 消融实验（对应报告五步）

只改几个开关即可得到对照：

| 方案 | 开关 |
|---|---|
| Baseline（无增强） | `--use_momentum False`，且把 `config.eot_*` 置 0、`num_eot=1`、单一 surrogate |
| + EOT | 打开 `eot_*` 范围 |
| + Momentum | `--use_momentum True` |
| + Ensemble | `--surrogates` 给多个模型 |
| TUPatch | 全部打开 |

## 指标定义

```
分母：干净图上被正确检测的 person 数（IoU≥0.5 且 score≥conf_threshold）
分子：加补丁后这些目标中"不再被正确检测"的数量
ASR  = 分子 / 分母
```

注意分母**不是**所有 GT person（否则模型本身漏检的目标会被误算成攻击成功）。

## YOLOv5 梯度路径（需要 ultralytics，接入模型后处理）

当前 `llm` 环境未装 `ultralytics`，torchvision 的 Faster R-CNN / RetinaNet 是可直接跑通的
参考路径。要加入 YOLOv5：

1. `pip install ultralytics`
2. 训练时 YOLO 需要走**原始检测头**才能把梯度回传到补丁（`model(img)` 的 predict 路径内部
   用了 `inference_mode`，梯度会被切断）。`detectors.py` 里的 `YoloDetector._forward_raw`
   目前是占位，需按你所用 ultralytics 版本的 `DetectionModel.model[-1]`（Detect 头）结构实现
   `_decode`：对每层 `train_out[i] ∈ [bs, na, ny, nx, 5+nc]` 解码
   `xy = (sigmoid(..)*2-0.5+grid)*stride`、`wh = (sigmoid(..)*2)^2*anchor`、
   `conf = sigmoid(obj) * sigmoid(cls[person])`，再把三尺度拼起来。
3. 评估路径 `_forward_predict` 无需梯度，已可直接使用。

## 依赖

见 `requirements.txt`。核心：`torch`、`torchvision`、`pillow`、`matplotlib`；
`ultralytics` 可选（仅 YOLO 用）。
