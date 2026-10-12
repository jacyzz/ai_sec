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
conda activate adv
cd /home/nfs/u2023-zlb/ai_sec
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
  --surrogates yolov5s fasterrcnn \
  --target person --epochs 30 --patch_size 64 --patch_ratio 0.1 \
  --out_dir outputs_tupatch
```

- `--surrogates` 可写 `yolov5s` / `fasterrcnn` / `retinanet` / `fcos`，
  或 `.pt` 权重路径（走 YOLO 封装）。权重首次使用会自动下载。
- 集成：`--surrogates yolov5s fasterrcnn` 即同时对两个模型取平均损失。

## 评估

```bash
python -m tupatch.main eval \
  --coco_root /path/to/coco \
  --surrogates yolov5s fasterrcnn \
  --blackbox retinanet \
  --out_dir outputs_tupatch
```

白盒/黑盒只是评估视角：参与训练的模型是白盒，`--blackbox` 里的模型**从未参与训练**，
其 ASR 就是迁移攻击成功率。评估会在 `outputs_tupatch/results.json` 输出各模型的
`clean_ap / clean_recall / attack_ap / attack_recall / asr`。
`eval` 会自动复用 `outputs_tupatch/config.json`（训练时保存），保证 surrogate/blackbox/target 与训练一致。

## 消融实验（四组，回答三个问题）

一键运行全部对照（Baseline / +EOT / +Static Ensemble / +Dynamic Ensemble）：

```bash
python run_ablation.py --coco_root /path/to/coco \
  --epochs 30 --num_train 2000 --num_test 500 --out_dir outputs_ablation
```

| 方案 | 配置 | 回答的问题 |
|---|---|---|
| Baseline | 单 YOLOv5s，无 EOT，无集成 | — |
| + EOT | 单 YOLOv5s + EOT | Q1: EOT 有用吗？ |
| + Static Ensemble | YOLOv5s+Faster R-CNN + EOT，等权重 | Q2: 多模型训练能提升迁移吗？ |
| + Dynamic Ensemble | 同上，但按「当前最难攻击的模型权重越大」动态加权 | Q3: 动态权重比静态平均好吗？ |

动态加权：`h_m = S_m(t)/S_m(0)`（当前 person 置信度 / 初始值），`w = softmax(h/τ)`，
τ 由 `config.ensemble_tau` 控制。开启 `--use_momentum` 已默认关闭（实验表明其对本任务无正收益）。

黑盒模型：RetinaNet + FCOS（从不参与训练），用于衡量迁移 ASR。

### 补丁参数化与尺度

`patch_size=64` 是**可学习纹理的基础分辨率**，不是补丁的显示尺寸。实际贴图边长由

```
side = sqrt(patch_ratio × bbox面积)
```

决定（`patch.py:_transform_patch`），64×64 的纹理被重采样到该尺寸后再贴到目标上。
因此**同一份纹理可以直接应用到不同尺度的目标，无需为每个目标单独训练补丁**。

这不是尺度不变（scale invariance）：纹理被放大/缩小时攻击效果会变化，
`run_patchsize.py` 正是在量化这一依赖。

## 数据划分

三个集合互不重叠：

| 集合 | 来源 | 数量 | 用途 |
|---|---|---|---|
| train | `train2017` 含 person 图，`entries[0:2000)` | 2000 | 训练补丁 |
| val | `train2017` 含 person 图，`entries[2000:2500)` | 500 | 开发期调参 |
| test | `val2017` 含 person 图，`[0:500)` | 500 | **final test，只在最后评估** |

`entries` 的顺序 = 标注 JSON 中 image 首次出现的顺序（dict 保序），**全程不排序**，
保证切片可复现。磁盘上 `train2017` 恰好是 `entries[0:2500]` 的连续前缀，
因此 train/val 两张切片的文件都在本地，且与历史实验的 train 集合完全一致。

## 补丁大小消融

固定 dynamic 方法，只改 `patch_ratio`（补丁面积 / person bbox 面积）：

```bash
python run_patchsize.py --coco_root /path/to/coco \
  --ratios 0.03 0.05 0.20 --epochs 30 --num_train 2000 --num_test 500
```

| ratio | 中位边长 | YOLOv5s (W) | Faster R-CNN (W) | RetinaNet (B) | FCOS (B) | 黑盒平均 |
|---|---|---|---|---|---|---|
| 3% | ~11 px | 0.234 | 0.277 | 0.285 | 0.275 | 0.280 |
| 5% | ~14 px | 0.296 | 0.306 | 0.327 | 0.308 | 0.318 |
| 10% | ~19 px | 0.670 | 0.492 | 0.516 | 0.490 | 0.503 |
| 20% | ~27 px | 0.935 | 0.783 | 0.797 | 0.803 | 0.800 |

非线性明显：5%→10% 跃升 +0.185，10%→20% 跃升 +0.282。**10% 是性价比拐点**。

注意 20% 档有 24.5% 的样本贴图边长超过 64px（纹理分辨率），属于上采样，
该档数值可能受纹理分辨率限制而非方法上限。

## 随机噪声对照

不训练，直接贴同尺寸随机噪声（3 个种子平均），给出「纯遮挡」基线：

```bash
python run_random_control.py --coco_root /path/to/coco \
  --ratios 0.03 0.05 0.10 0.20 --num_test 500 --out_dir outputs_random
```

| ratio | 优化补丁（黑盒平均） | 随机噪声 | 增益 |
|---|---|---|---|
| 3% | 0.280 | 0.108 | +0.172 |
| 5% | 0.318 | 0.133 | +0.185 |
| 10% | 0.503 | 0.187 | +0.316 |
| 20% | 0.800 | 0.287 | +0.513 |

随机噪声的 ASR 随面积近似线性增长（遮挡效应），而优化补丁是超线性的，
增益随面积扩大——这证明补丁的收益来自对抗优化而非遮挡。

## 多种子重复实验

对比 Static / Dynamic Ensemble 在 3 个随机种子下的 mean ± std：

```bash
python run_seeds.py --coco_root /path/to/coco --seeds 42 43 44 \
  --methods static dynamic --epochs 30 --out_dir outputs_seeds
```

`run_seeds.py` 采用上面的 train/val/test 三划分：训练用 train，评估同时报 val 与 test。

## 指标定义

```
分母：干净图上被正确检测的 person 数（IoU≥0.5 且 score≥conf_threshold）
分子：加补丁后这些目标中"不再被正确检测"的数量
ASR  = 分子 / 分母
```

注意分母**不是**所有 GT person（否则模型本身漏检的目标会被误算成攻击成功）。

## YOLOv5（官方 ultralytics/yolov5 仓库）

`YoloDetector` 直接调用仓库根目录的 `yolov5/`（本地 clone），无需 `pip install ultralytics` 包：

- 训练走**原始检测头**（可微）：临时把 `Detect` 头置为 training，拿到原始
  `[bs, na, ny, nx, 5+nc]` 特征后手动解码 `xy/wh/obj/cls`，再取
  `conf = sigmoid(obj) * sigmoid(cls[person])` 作为 person 置信度，梯度可回传到补丁。
- 评估走标准 inference 路径（letterbox → 模型 → NMS → 缩放回原图坐标）。
- 权重 `yolov5s.pt` 首次使用自动下载到 `yolov5/` 目录。
- `--yolov5_repo` 可指定仓库路径（默认自动探测 `<repo>/yolov5`），`--yolo_imgsz` 指定输入边长。

## 依赖

见 `requirements.txt`。核心：`torch`、`torchvision`、`pillow`、`matplotlib`；
YOLOv5 仓库自身的依赖（`opencv-python`、`PyYAML`、`ultralytics` 等）按其 `yolov5/requirements.txt` 安装。
