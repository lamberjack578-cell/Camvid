# 归档：工程改造前的 2 epoch 冒烟产物

归档时间：2026-11-24
原始时间戳：2026-11-24 01:04

这些文件由**旧版** `src/train_seg.py` 在 E0 工程改造之前生成，属于 2 epoch 冒烟测试，**不是正式 E0 结果**。

| 文件 | 内容 | 原路径 |
|---|---|---|
| `metrics/E0_original.csv` | 2 epoch 结果：mIoU 0.4226，pixel_acc 0.8659，11 类 IoU | `results/metrics/E0_original.csv` |
| `metrics/train_log.csv` | 仅 epoch 1–2 的日志 | `results/metrics/train_log.csv` |
| `weights/unet_resnet34_camvid.pth` | 2 epoch 时 best 权重（仅 state_dict，94 MB） | `weights/unet_resnet34_camvid.pth` |

旧版脚本把这些路径写死，每次运行都会覆盖；改造后新流程使用 `results/<exp.name>/` 独立目录，不再读取上述位置。

正式 E0 结果见 `results/E0/`：

- 50 epoch，最优 epoch 48
- Best Val mIoU 0.7554
- Test mIoU 0.6687，Test Pixel Accuracy 0.9038

本目录仅用于留存历史，请勿将其中的数值当作 E0 基线。
