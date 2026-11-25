import numpy as np, cv2
from pathlib import Path

root = Path("data/camvid")
for split in ["train", "val", "test"]:
    imgs = sorted((root / split).glob("*.png"))
    labs = sorted((root / f"{split}annot").glob("*.png"))
    print(f"[{split}] 图像 {len(imgs)} 张, 标注 {len(labs)} 张")
    assert len(imgs) == len(labs), "图像与标注数量不一致"

# 抽查第一张训练图
img = cv2.imread(str(sorted((root / "train").glob("*.png"))[0]))
lab = cv2.imread(str(sorted((root / "trainannot").glob("*.png"))[0]), cv2.IMREAD_UNCHANGED)
print("图像 shape/dtype:", img.shape, img.dtype)
print("标注 shape/dtype:", lab.shape, lab.dtype)

# 统计整个训练集的类别像素占比
counts = np.zeros(256, dtype=np.int64)
for p in (root / "trainannot").glob("*.png"):
    l = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    counts += np.bincount(l.ravel(), minlength=256)
for k in np.nonzero(counts)[0]:
    print(f"标签值 {k:3d}: {counts[k] / counts.sum():.3%}")