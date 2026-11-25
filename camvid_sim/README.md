# CamVid 语义分割实验（camvid_sim）

CamVid 街景语义分割基准实验工程。当前阶段：**E0（原始图像基线）已完成 50 epoch 正式训练，测试集 mIoU 0.6687**。

运行环境（实测）：

| 项 | 值 |
|---|---|
| Python | 3.13.2 |
| PyTorch | 2.15.0+cu132 |
| CUDA | 13.2 |
| cuDNN | 92600 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| 解释器 | `/d/camvid_env/Scripts/python.exe` |

---

## E0 实验定义（已锁定，不得随意改动）

- 模型：U-Net + ResNet34，ImageNet 预训练 Encoder
- 优化器：Adam，lr = 1e-4，batch size = 4
- 损失：CrossEntropy + Dice（均忽略 VOID）
- 数据增强：仅随机裁剪 320×448 + 水平翻转
- 数据划分：train / val / test = 367 / 101 / 233（官方 SegNet CamVid 划分）
- 类别：11 类（0–10），VOID = 11
- 训练 50 epoch，按 Val mIoU 保存最优权重
- **不加 LR Scheduler，不加 Early Stopping**

---

## E0 正式实验结果（2026-11-24）

实验目录：`results/E0/`，命令 `python src/train_seg.py --config configs/default.yaml`，seed=42。

| 项 | 值 |
|---|---|
| 训练轮数 | 50 epoch（无断点恢复） |
| 总耗时 | 639.8 s（约 10.7 min，约 12.8 s/epoch） |
| 最优 epoch | 48 / 50 |
| Best Val mIoU | 0.7554 |
| **Test mIoU** | **0.6687** |
| Test Pixel Accuracy | 0.9038 |

测试集各类别指标（由 `metrics/test_confusion_matrix.csv` 计算）：

| 类别 | IoU | Recall | Precision |
|---|---:|---:|---:|
| Sky | 0.9220 | 0.9577 | 0.9612 |
| Building | 0.8220 | 0.9054 | 0.8992 |
| Pole | 0.3167 | 0.4285 | 0.5483 |
| Road | 0.9147 | 0.9780 | 0.9339 |
| Pavement | 0.7188 | 0.8068 | 0.8683 |
| Tree | 0.7578 | 0.8677 | 0.8568 |
| SignSymbol | 0.4411 | 0.5639 | 0.6696 |
| Fence | 0.4415 | 0.6221 | 0.6034 |
| Car | 0.8216 | 0.8585 | 0.9503 |
| Pedestrian | 0.5788 | 0.7222 | 0.7445 |
| Bicyclist | 0.6211 | 0.7111 | 0.8306 |

观察：

- **Val 与 Test 差距明显**：best val mIoU 0.7554，test mIoU 0.6687，差 0.0867。val 只有 0016E5 序列、101 张，难度与 test 不同，报告 E0 时应以 test 为准。
- **弱类别**：Pole（0.3167）、SignSymbol（0.4411）、Fence（0.4415）最差；混淆矩阵显示它们主要被误判为 Building（分别占真实类的 31.95%、35.67%、23.43%），细长/小目标在 ResNet34 下采样后丢失严重。
- **Pavement ↔ Road 互混**：17.09% 的 Pavement 被判为 Road，是第二大类错误来源。
- **训练曲线**：无 LR 衰减，val mIoU 在 0.65–0.755 间波动；最优出现在 epoch 48（0.7554），最后一轮 epoch 50 为 0.7472；最小 val loss 在 epoch 38（0.4101），与最优 mIoU 不同步，属 CE+Dice 下的正常现象。模型选择按 Val mIoU，符合 E0 定义。

产物：

- `metrics/test_metrics.csv`：mIoU、pixel_acc、11 类 IoU，best_epoch=48，best_val_miou=0.7554
- `metrics/test_confusion_matrix.csv`：11×11 混淆矩阵（3871.9 万标注像素，VOID 已排除）
- `figures/test/`：12 张测试图 × 原图/GT/预测/叠加 = 48 个文件
- `weights/best_model.pth`（epoch 48，94 MB）、`weights/checkpoint_last.pth`（epoch 50，280 MB）
- 目录总计约 380 MB

---

## 快速开始

所有命令在项目根目录 `D:/camvid_sim` 执行。

```bash
# 冒烟测试（1 epoch，写入 results/smoke/，不评估测试集，不碰正式产物）
python src/train_seg.py --smoke

# 正式 E0 训练（50 epoch，产物写入 results/E0/）
python src/train_seg.py --config configs/default.yaml

# 中断恢复
python src/train_seg.py --config configs/default.yaml --resume

# 独立评估（不训练，重新生成测试集指标/混淆矩阵/可视化）
python src/train_seg.py --eval-only --run-dir results/E0

# 非正式检查可评估 val
python src/train_seg.py --eval-only --run-dir results/E0 --eval-split val
```

注意事项：

- 正式实验目录 `results/<exp.name>/` 已存在且非空时，脚本会拒绝启动，除非使用 `--resume`，防止覆盖 E0。
- `--smoke` 自动使用 `results/smoke/<exp.name>_<时间戳>/`，每次独立。
- `--skip-test` 可让正式目录的训练暂不评估测试集（工程调试用；正式 E0 不要加）。
- 测试集正式评估只在训练结束、模型冻结后执行一次；`--eval-only` 用于显式重新生成评估产物。
- 续训时 seed 与 checkpoint 不一致会中止；其他配置与快照不一致只告警。

---

## 目录结构

```
camvid_sim/
├── configs/
│   └── default.yaml              # 实验配置（含 exp.name、seed、train、eval）
├── data/camvid/                  # train/val/test + *annot + 官方划分 txt
├── src/
│   ├── dataset.py                # Dataset、标准化、调色板、配对断言
│   ├── metrics.py                # 混淆矩阵 / 各类 IoU
│   ├── inspect_data.py           # 数据抽查脚本
│   └── train_seg.py              # 训练 / 恢复 / 评估入口
├── results/
│   ├── E0/                       # 正式实验产物（已完成，50 epoch）
│   ├── smoke/                    # 冒烟测试独立产物
│   ├── figures/                  # 早期 dataset_check.png 调试图
│   └── legacy_smoke_20261124_0104/  # 工程改造前 2 epoch 冒烟产物（已归档）
└── README.md
```

单个实验目录：

```
results/<exp>/
├── config.yaml                   # 本次实验配置快照
├── env.json                      # 运行时间、版本、GPU、seed、argv、耗时
├── weights/
│   ├── best_model.pth            # val mIoU 最优权重（含 epoch / val_miou）
│   └── checkpoint_last.pth       # 完整断点（模型/优化器/scaler/RNG/epoch/best）
├── metrics/
│   ├── train_log.csv             # 每 epoch: epoch, seed, train_loss, val_loss, val_mIoU, val_pixel_acc, lr, epoch_seconds
│   ├── {split}_metrics.csv       # mIoU、pixel_acc、11 类 IoU
│   └── {split}_confusion_matrix.csv
└── figures/<split>/              # 每样本: _1_image, _2_gt, _3_pred, _4_overlay
```

`eval.vis_samples` 控制可视化样本数（默认 12，设为 0 表示全部 233 张测试图）。

---

## 变更日志

### 2026-11-24 — E0 正式训练前工程整理

#### 背景

E0 的 2 epoch 冒烟测试通过后，对 `src/train_seg.py` 及相关代码做了审查，发现以下问题：

- 只保存最优权重，无断点恢复；中断即从头重训；
- 权重、日志、指标均写入固定路径，后续 E1/E2 会覆盖 E0；
- `train_log.csv` 每次运行以 `w` 模式清空；
- cudnn 确定性未开启，DataLoader 随机性未按 num_workers 处理；
- 混淆矩阵只存在内存、测试集无可视化；
- 无独立评估入口，不重训无法复看结果；
- 图像与标注仅按数量断言，依赖 `sorted()` 隐式配对。

本次只做工程改造，**未改模型结构、损失函数、数据划分与任何 E0 物理实验参数**。

#### 文件修改

| 文件 | 修改内容 |
|---|---|
| `src/train_seg.py` | 重写训练脚本的工程层：完整 checkpoint / 恢复、实验目录隔离、配置与环境快照、确定性随机、评估产物落盘、`--eval-only`、`--smoke`、`--skip-test`、`--run-dir`、`--weights`、`--eval-split` |
| `src/dataset.py` | 新增图像/标注文件名逐一配对断言；`PALETTE` 从 `__main__` 提升到模块级，供评估可视化复用 |
| `configs/default.yaml` | 新增 `exp.name`、`exp.description`、`eval.vis_samples`；移除被运行目录取代的 `train.weights_path` |

#### 实现明细

1. **完整 Checkpoint**
   - 每 epoch 原子写入 `weights/checkpoint_last.pth`（`.tmp` + `os.replace`）。
   - 保存 `model_state_dict`、`optimizer_state_dict`、`scaler_state_dict`、`epoch`、`best_val_miou`、`seed`、`train_generator_state`、Python/NumPy/Torch/CUDA RNG 状态。
   - `--resume` 恢复以上全部状态，并从下一 epoch 继续；恢复时截断日志中 checkpoint 之后多写的行。
   - seed 与 checkpoint 不一致时中止训练。

2. **实验产物隔离**
   - 正式实验：`results/<exp.name>/`；E1/E2 只需改 `exp.name` 或 `--run-dir`。
   - 目录非空时拒绝启动（除非 `--resume`）。
   - 保存 `config.yaml` 原文快照和 `env.json`（启动/结束时间、总耗时、Python/PyTorch/CUDA/cuDNN 版本、GPU 型号、seed、argv、cwd）。

3. **随机性控制**
   - 固定 Python、NumPy、PyTorch、CUDA 种子。
   - `torch.backends.cudnn.deterministic = True`，`torch.backends.cudnn.benchmark = False`。
   - `torch.use_deterministic_algorithms(True, warn_only=True)`。
   - `num_workers > 0` 时通过 `worker_init_fn` 基于 `torch.initial_seed()` 派生 worker 种子；train DataLoader 使用固定种子的 `torch.Generator`，其状态纳入 checkpoint。
   - seed 写入 `config.yaml`、`env.json` 以及 `train_log.csv` 每行。

4. **评估产物**
   - `metrics/{split}_metrics.csv`：exp、split、seed、best_epoch、best_val_miou、mIoU、pixel_acc、11 类 IoU。
   - `metrics/{split}_confusion_matrix.csv`：行=真实类，列=预测类。
   - `figures/{split}/`：原图、GT、预测、叠加图，预测裁回原始分辨率。
   - 正式训练结束时测试集只评估一次；若 `test_metrics.csv` 已存在则跳过，避免反复查看测试集。

5. **独立评估**
   - `--eval-only` 读取实验目录内 `config.yaml` 快照与 `weights/best_model.pth`，无需重训。
   - 默认评估 test，支持 `--eval-split val` 做非正式检查，支持 `--weights` 指定其他权重。

6. **冒烟模式**
   - `--smoke` 写入 `results/smoke/<exp.name>_<时间戳>/`，每次唯一，不覆盖正式实验。
   - 未指定 `--epochs` 时默认 1 epoch；smoke 不评估测试集。
   - `--smoke` 与 `--resume` / `--eval-only` 互斥。

7. **数据配对断言**
   - 数量或文件名集合不一致时抛出明确错误，并列出最多 5 个不匹配项。

#### 验证记录

以下测试全部使用 `--smoke` 或 `--skip-test` 的独立目录，未产生正式 E0 产物，未评估正式测试集（`--eval-only` 验证使用 val split）。

| # | 测试项 | 命令 / 操作 | 结果 |
|---|---|---|---|
| 1 | 语法与导入 | `py_compile` 三个源文件 | 通过 |
| 2 | 数据配对断言 | 实例化 train/val/test Dataset | 367/101/233 全部通过 |
| 3 | 1 epoch 冒烟训练 | `--smoke --epochs 1` | train_loss 2.1743，val_loss 1.4443，val_mIoU 0.4298，val_pixel_acc 0.8670，13.9 s/epoch，峰值 1.04 GB |
| 4 | 产物完整性 | 检查 smoke 目录 | `config.yaml`、`env.json`、`train_log.csv`、`best_model.pth`、`checkpoint_last.pth` 均生成 |
| 5 | checkpoint 内容 | 加载 `checkpoint_last.pth` | epoch=2（续训测试后）、best=0.4485、seed=42、优化器 140 项状态、scaler scale=65536、generator state、CUDA RNG 均在 |
| 6 | 中断恢复 | 独立目录先 1 epoch，再 `--resume --epochs 2` | 从 epoch 2 正确继续，best 恢复为 0.4298 |
| 7 | 恢复一致性 | 对比“1+1 续训”与“连续 2 epoch” | epoch-2 指标逐位一致：train_loss 1.4062，val_loss 1.1915，val_mIoU 0.4485，val_pixel_acc 0.8725 |
| 8 | 目录覆盖保护 | 对非空目录重复启动（不带 `--resume`） | 被拒绝，原有文件未改动 |
| 9 | 完成后续训 | 对已完成 2 epoch 的目录 `--resume --epochs 2` | 正确跳过训练，未报错，env 记录 resumed=true |
| 10 | 独立评估 | `--eval-only --eval-split val` | val mIoU 0.4485，与训练日志一致；指标 CSV、混淆矩阵 CSV、48 张可视化（12 样本 × 4）生成 |
| 11 | 可视化正确性 | 抽查 overlay 与 GT 图 | 颜色与轮廓对齐正确 |
| 12 | 正式目录隔离 | 检查 `results/E0` | 不存在，冒烟/测试未污染正式路径 |
| 13 | 参数互斥 | `--smoke --resume` | 正确报错退出 |
| 14 | 目录体积 | smoke 目录 | checkpoint 约 280 MB，best_model 约 94 MB，单实验约 381 MB（epoch 覆盖写，不随轮数增长） |

冒烟产物保留在：`results/smoke/E0_20261124_012720/`。

#### 未做 / 遗留问题

1. **CE 反向无确定性 CUDA 实现**：会告警 `nll_loss2d_forward_out_cuda_template`。cudnn 已确定性化，但 PyTorch 自身无法保证该算子逐位复现；实测续训与连续训练一致。
2. **正式训练结束会自动评估一次测试集**（`--skip-test` 可跳过）；同一实验目录内不会重复评估，`--eval-only` 是显式重新生成并会覆盖评估 CSV/图。
3. **旧代码遗留产物**：`results/metrics/E0_original.csv`、`results/metrics/train_log.csv`、`weights/unet_resnet34_camvid.pth` 是 01:04 那次 2 epoch 冒烟的产物，不属于新流程；已于 2026-11-24 归档至 `results/legacy_smoke_20261124_0104/`（内含 NOTE.md 说明）。
4. **无 LR scheduler / 早停**：这是 E0 定义要求；若 50 epoch 欠拟合，需要单独立项，不能改动 E0 口径。
5. **默认只可视化 12 张**测试图；全量需将 `eval.vis_samples` 设为 0。
6. **官方 CamVid 划分的固有属性**：train/test 共享 `0001TP` 视频段（训练段尾 8520 与测试段首 8550 仅隔 30 帧），train/val 共享 `0016E5`（val 7959–8159，两侧约 1 秒间隔）。E0 绝对值会偏乐观，但与文献使用同一官方划分，未改动。
7. **控制台中文在 Git Bash 下乱码**：终端编码问题，CSV/JSON 均为 UTF-8；可设置 `PYTHONIOENCODING=utf-8`。
8. **续训配置校验**：seed 不一致中止；其余配置与快照不一致仅告警并继续。

#### 当时状态

- 工程整理完成时正式 E0 尚未启动；随后于同日完成，见下一条记录。
- 旧版遗留产物当时散落在 `results/metrics/`、`weights/`，后于同日归档至 `results/legacy_smoke_20261124_0104/`。

---

### 2026-11-24 — E0 正式训练完成（50 epoch）

#### 执行信息

| 项 | 值 |
|---|---|
| 命令 | `python src/train_seg.py --config configs/default.yaml` |
| 实验目录 | `results/E0/` |
| 启动 / 结束 | 01:49:36 / 02:00:15 |
| 总耗时 | 639.8 s（约 10.7 min） |
| 模式 | formal，未使用 `--resume` / `--smoke` / `--skip-test` / `--eval-only` |
| seed | 42 |
| epoch | 50（每 epoch 约 12.8 s） |

#### 结果

| 指标 | 值 |
|---|---:|
| Best epoch | 48 / 50 |
| Best Val mIoU | 0.7554 |
| **Test mIoU** | **0.6687** |
| Test Pixel Accuracy | 0.9038 |

测试集 11 类 IoU：

```
Sky        0.9220    Building   0.8220    Pole       0.3167
Road       0.9147    Pavement   0.7188    Tree       0.7578
SignSymbol 0.4411    Fence      0.4415    Car        0.8216
Pedestrian 0.5788    Bicyclist  0.6211
```

关键观察：

- Val（0.7554）与 Test（0.6687）相差 0.0867；val 仅含 0016E5 序列 101 张，不能代表 test，正式汇报以 test 为准。
- 最差类为 Pole 0.3167、SignSymbol 0.4411、Fence 0.4415，主要被误判为 Building（占各类真实像素的 31.95% / 35.67% / 23.43%）。
- 次要错误来源：17.09% 的 Pavement 被判成 Road。
- 无 LR 衰减，val mIoU 在 0.65–0.755 波动；最优 epoch 48，末轮 epoch 50 为 0.7472；最小 val loss 出现在 epoch 38（0.4101），与最优 mIoU 不同步，属正常现象。按 Val mIoU 选模型符合 E0 定义。
- 主体类别（Sky/Road/Building/Car）IoU 均 ≥ 0.82，说明训练流程与评估链路正常。

#### 产物

- `metrics/train_log.csv`：50 行完整记录（epoch、seed、train_loss、val_loss、val_mIoU、val_pixel_acc、lr、epoch_seconds）
- `metrics/test_metrics.csv`：Test mIoU / pixel_acc / 11 类 IoU（best_epoch=48，best_val_miou=0.7554）
- `metrics/test_confusion_matrix.csv`：11×11，3871.9 万标注像素，VOID 已排除
- `figures/test/`：12 张测试图 × 4（原图/GT/预测/叠加）= 48 个文件
- `weights/best_model.pth`：epoch 48，94 MB
- `weights/checkpoint_last.pth`：epoch 50，280 MB（完整断点，可 `--resume`）
- `config.yaml`、`env.json`：配置与运行环境快照
- 目录总计约 380 MB

#### 说明

- 本阶段**未修改任何代码或配置**，只运行正式训练并更新本日志。
- 测试集仅在训练结束、模型冻结后评估一次，未参与训练或模型选择。
- 旧版遗留产物（原 `results/metrics/`、`weights/unet_resnet34_camvid.pth`）已归档至 `results/legacy_smoke_20261124_0104/`；后续 E1 应新建 `exp.name` 或 `--run-dir`，不会覆盖 `results/E0/`。
- `checkpoint_last.pth` 对应 epoch 50，但训练已完成；若要基于 E0 继续训练，应先明确是否改变实验定义，不能默认续训。

---

### 2026-11-24 — 旧版冒烟产物归档

- 将旧代码 2 epoch 冒烟产物移至 `results/legacy_smoke_20261124_0104/`：
  - `metrics/E0_original.csv`
  - `metrics/train_log.csv`
  - `weights/unet_resnet34_camvid.pth`
- 归档目录内新增 `NOTE.md`，说明来源、原始路径及“不得作为 E0 基线”的提示。
- 清空后的 `results/metrics/`、`weights/` 空目录已删除。
- 同步更新 README 中相关引用；未改动代码、配置或正式实验结果。

---

## 当前状态

- **E0 已完成**：Test mIoU 0.6687，Pixel Accuracy 0.9038，最优 epoch 48。
- 正式产物位于 `results/E0/`，可直接用 `--eval-only --run-dir results/E0` 复现评估。
- 旧版 2 epoch 冒烟产物已归档至 `results/legacy_smoke_20261124_0104/`，不影响 E0。
- 下一步若进入 E1/E2，应使用新的 `exp.name`，保持 E0 产物不被覆盖。
