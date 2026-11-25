from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_NAMES = ["Sky", "Building", "Pole", "Road", "Pavement", "Tree",
               "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist"]
NUM_CLASSES = 11
VOID = 11  # 未标注像素，不参与损失与指标

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CamVid 调色板，索引 0..10 为类别，索引 11 为 VOID
PALETTE = np.array([[128,128,128],[128,0,0],[192,192,128],[128,64,128],[60,40,222],
                    [128,128,0],[192,128,128],[64,64,128],[64,0,128],[64,64,0],
                    [0,128,192],[0,0,0]], dtype=np.uint8)


class CamVidDataset(Dataset):
    def __init__(self, root="data/camvid", split="train", crop_size=(320, 448)):
        root = Path(root)
        self.img_paths = sorted((root / split).glob("*.png"))
        self.msk_paths = sorted((root / f"{split}annot").glob("*.png"))
        img_names = [p.name for p in self.img_paths]
        msk_names = [p.name for p in self.msk_paths]
        assert len(img_names) == len(msk_names) > 0, \
            f"{split}: 数据缺失或数量不一致 (图像 {len(img_names)} / 标注 {len(msk_names)})"
        # 文件名必须逐一对应，否则 sorted 后按下标配对会静默错位
        if img_names != msk_names:
            only_img = sorted(set(img_names) - set(msk_names))[:5]
            only_msk = sorted(set(msk_names) - set(img_names))[:5]
            raise AssertionError(
                f"{split}: 图像与标注文件名不匹配，仅图像有 {only_img}，仅标注有 {only_msk}")
        self.train = (split == "train")
        self.crop_h, self.crop_w = crop_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img = cv2.imread(str(self.img_paths[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # OpenCV 默认 BGR，预训练模型要 RGB
        msk = cv2.imread(str(self.msk_paths[i]), cv2.IMREAD_UNCHANGED)

        if self.train:
            img, msk = self._random_crop(img, msk)
            if torch.rand(1).item() < 0.5:                   # 水平翻转，图和标签必须同步
                img, msk = img[:, ::-1], msk[:, ::-1]
        else:
            img, msk = self._pad_to_32(img, msk)

        img = np.ascontiguousarray(img).astype(np.float32) / 255.0   # [0,1]
        msk = np.ascontiguousarray(msk).astype(np.int64)
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(msk)  # (3,H,W), (H,W)

    def _random_crop(self, img, msk):
        h, w = msk.shape
        top = torch.randint(0, h - self.crop_h + 1, (1,)).item()
        left = torch.randint(0, w - self.crop_w + 1, (1,)).item()
        sl = (slice(top, top + self.crop_h), slice(left, left + self.crop_w))
        return img[sl], msk[sl]

    @staticmethod
    def _pad_to_32(img, msk):
        h, w = msk.shape
        ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32      # 360 -> 补 24 行变 384；480 已是 32 的倍数
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), constant_values=0)
        msk = np.pad(msk, ((0, ph), (0, pw)), constant_values=VOID)
        return img, msk


def normalize(x):
    """x: (B,3,H,W)，[0,1] 浮点。在送入模型前调用，之后的阶段会把过曝/器件模型插在它前面。"""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# ---------- 自检：python src/dataset.py ----------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    train_set, val_set = CamVidDataset(split="train"), CamVidDataset(split="val")
    print("训练/验证样本数:", len(train_set), len(val_set))

    loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
    x, y = next(iter(loader))
    print("训练 batch 图像:", tuple(x.shape), x.dtype, f"范围[{x.min():.2f}, {x.max():.2f}]")
    print("训练 batch 标签:", tuple(y.shape), y.dtype, "出现的值:", torch.unique(y).tolist())
    print("训练 batch 每轮迭代数:", len(loader))

    xv, yv = val_set[0]
    print("验证单张:", tuple(xv.shape), tuple(yv.shape), "| 补出的底部24行全是void:", bool((yv[360:] == VOID).all()))

    fig, axes = plt.subplots(2, 4, figsize=(16, 5))
    for i in range(4):
        axes[0, i].imshow(x[i].permute(1, 2, 0).numpy()); axes[0, i].axis("off")
        axes[1, i].imshow(PALETTE[y[i].numpy()]); axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig("results/figures/dataset_check.png", dpi=120)
    print("已保存 results/figures/dataset_check.png")