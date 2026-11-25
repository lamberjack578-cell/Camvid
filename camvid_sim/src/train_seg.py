"""CamVid 语义分割 E0 基准的训练 / 评估脚本。

E0 实验定义保持不变：
    U-Net + ResNet34(ImageNet) | Adam lr=1e-4 | batch=4 | CE+Dice
    仅随机裁剪 + 水平翻转 | train/val/test = 367/101/233 | 无 scheduler / 无早停
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from dataset import (CamVidDataset, normalize, NUM_CLASSES, VOID, CLASS_NAMES, PALETTE)
from metrics import confusion_matrix, iou_from_cm


LOG_HEADER = ["epoch", "seed", "train_loss", "val_loss", "val_mIoU", "val_pixel_acc", "lr", "epoch_seconds"]


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=VOID)
        self.dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=VOID)

    def forward(self, logits, y):
        return self.ce(logits, y) + self.dice(logits, y)


# ---------------------------------------------------------------- 随机性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 让 cuDNN 走确定性算法，并关闭自动调优（两者共同保证可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 部分 CUDA 算子没有确定性实现，warn_only 只告警不崩溃
    torch.use_deterministic_algorithms(True, warn_only=True)


def _worker_init(worker_id):
    """num_workers > 0 时给每个 worker 一个可复现的种子。

    DataLoader 每轮会从主进程 generator 派生 base_seed，torch.initial_seed()
    拿到的是本轮该 worker 的种子，因此每轮增强不同、但整条序列可复现。
    """
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


# ---------------------------------------------------------------- 评估
@torch.no_grad()
def evaluate_split(model, loader, device, loss_fn=None):
    model.eval()
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(normalize(x))
        logits = logits.float()
        if loss_fn is not None:
            total_loss += loss_fn(logits, y).item() * x.size(0)
            n += x.size(0)
        cm += confusion_matrix(logits.argmax(1), y)
    iou = iou_from_cm(cm)
    return dict(miou=torch.nanmean(iou).item(), iou=iou.cpu().numpy(),
                acc=(cm.diag().sum() / cm.sum()).item(), loss=total_loss / max(n, 1),
                cm=cm.cpu().numpy())


# ---------------------------------------------------------------- 文件工具
def atomic_torch_save(obj, path):
    """先写临时文件再原子替换，避免中断时留下损坏的 checkpoint。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_log_header(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(LOG_HEADER)


def append_log(path, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def truncate_log(path, upto_epoch):
    """恢复训练时丢弃 checkpoint 之后的日志行，避免重复记录。"""
    if not path.exists():
        write_log_header(path)
        return
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        write_log_header(path)
        return
    header, body = rows[0], rows[1:]
    kept = [r for r in body if r and int(r[0]) <= upto_epoch]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(kept)


def make_dirs(run_dir):
    for sub in ("weights", "metrics", "figures"):
        (Path(run_dir) / sub).mkdir(parents=True, exist_ok=True)


def ensure_fresh_run_dir(run_dir):
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(
            f"[中止] 实验目录已存在且非空: {run_dir}\n"
            "为避免覆盖已有实验结果，请更换配置中的 exp.name 或 --run-dir；\n"
            "若要在该目录继续训练，请使用 --resume。")
    run_dir.mkdir(parents=True, exist_ok=True)


def copy_config(config_path, run_dir):
    shutil.copyfile(config_path, Path(run_dir) / "config.yaml")


def check_config_snapshot(run_dir, cfg):
    snap_path = Path(run_dir) / "config.yaml"
    if not snap_path.exists():
        print(f"[警告] 未找到配置快照 {snap_path}，无法核对")
        return
    snap = yaml.safe_load(snap_path.read_text(encoding="utf-8"))
    if snap != cfg:
        print("[警告] 当前配置与实验快照不一致，将继续使用当前配置，请确认这是预期行为")


def write_env(run_dir, seed, argv, started_at):
    env = {
        "started_at": datetime.fromtimestamp(started_at).isoformat(timespec="seconds"),
        "seed": seed,
        "argv": argv,
        "cwd": str(Path.cwd()),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
    }
    (Path(run_dir) / "env.json").write_text(
        json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")


def finish_env(run_dir, session_started_at, extra):
    path = Path(run_dir) / "env.json"
    env = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    try:                                    # 恢复训练时 elapsed 从实验最初启动算起
        t0 = datetime.fromisoformat(env["started_at"]).timestamp()
    except (KeyError, ValueError):
        t0 = session_started_at
    env.update({"finished_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": round(time.time() - t0, 1),
                "last_session_seconds": round(time.time() - session_started_at, 1)})
    env.update(extra)
    path.write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- checkpoint
def save_checkpoint(path, model, optimizer, scaler, epoch, best, seed, train_gen):
    atomic_torch_save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "best_val_miou": best,
        "seed": seed,
        "train_generator_state": train_gen.get_state(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }, path)


def restore_checkpoint(ckpt, model, optimizer, scaler, train_gen, device):
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if "train_generator_state" in ckpt:
        train_gen.set_state(ckpt["train_generator_state"].cpu())
    rng = ckpt.get("rng")
    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
    return int(ckpt["epoch"]), float(ckpt["best_val_miou"])


# ---------------------------------------------------------------- 评估产物
def save_split_metrics(res, path, exp_name, split, seed, best_epoch, best_val_miou):
    header = (["exp", "split", "seed", "best_epoch", "best_val_miou", "mIoU", "pixel_acc"]
              + [f"IoU_{n}" for n in CLASS_NAMES])
    row = ([exp_name, split, seed, best_epoch, best_val_miou,
            f"{res['miou']:.4f}", f"{res['acc']:.4f}"]
           + [f"{v:.4f}" for v in res["iou"]])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow(row)


def save_confusion_matrix(cm, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + CLASS_NAMES)
        for name, row in zip(CLASS_NAMES, cm):
            w.writerow([name] + [int(v) for v in row])


@torch.no_grad()
def save_visualizations(model, dataset, device, out_dir, max_samples):
    """保存 split 的原图 / GT / 预测 / 叠加图（预测裁回原始分辨率）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(dataset) if max_samples in (None, 0) else min(int(max_samples), len(dataset))
    model.eval()
    for i in range(n):
        x, _ = dataset[i]
        img_bgr = cv2.imread(str(dataset.img_paths[i]))
        gt = cv2.imread(str(dataset.msk_paths[i]), cv2.IMREAD_UNCHANGED)
        h, w = gt.shape
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(normalize(x.unsqueeze(0).to(device)))
        pred = logits.argmax(1)[0, :h, :w].cpu().numpy()

        gt_color = PALETTE[np.clip(gt, 0, NUM_CLASSES)].astype(np.uint8)     # 11 -> 黑色
        pred_color = PALETTE[pred].astype(np.uint8)
        overlay = cv2.addWeighted(img_bgr, 0.55,
                                  cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR), 0.45, 0)

        stem = dataset.img_paths[i].stem
        cv2.imwrite(str(out_dir / f"{stem}_1_image.png"), img_bgr)
        cv2.imwrite(str(out_dir / f"{stem}_2_gt.png"), cv2.cvtColor(gt_color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{stem}_3_pred.png"), cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{stem}_4_overlay.png"), overlay)


def evaluate_and_save(model, loader, dataset, device, run_dir, split, exp_name,
                      seed, vis_samples, best_epoch=None, best_val_miou=None):
    res = evaluate_split(model, loader, device)
    save_split_metrics(res, Path(run_dir) / "metrics" / f"{split}_metrics.csv",
                       exp_name, split, seed, best_epoch, best_val_miou)
    save_confusion_matrix(res["cm"], Path(run_dir) / "metrics" / f"{split}_confusion_matrix.csv")
    save_visualizations(model, dataset, device, Path(run_dir) / "figures" / split, vis_samples)
    print(f"\n[{split} 评估] mIoU {res['miou']:.4f} | 像素准确率 {res['acc']:.4f}")
    for n, v in zip(CLASS_NAMES, res["iou"]):
        print(f"  {n:11s} IoU {v:.4f}")
    print(f"  指标: {Path(run_dir) / 'metrics' / f'{split}_metrics.csv'}")
    print(f"  混淆矩阵: {Path(run_dir) / 'metrics' / f'{split}_confusion_matrix.csv'}")
    print(f"  可视化: {Path(run_dir) / 'figures' / split}")
    return res


# ---------------------------------------------------------------- 模型
def build_model():
    return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                    in_channels=3, classes=NUM_CLASSES)


def resolve_run_dir(cfg, args):
    if args.run_dir:
        return Path(args.run_dir)
    exp_name = cfg.get("exp", {}).get("name", "E0")
    if args.smoke:
        ts = time.strftime("%Y%m%d_%H%M%S")
        return Path("results") / "smoke" / f"{exp_name}_{ts}"
    return Path("results") / exp_name


# ---------------------------------------------------------------- 评估入口
def run_eval_only(cfg, run_dir, args):
    seed = int(cfg["seed"])
    exp_name = cfg.get("exp", {}).get("name", "E0")
    set_seed(seed)
    assert torch.cuda.is_available(), "需要 GPU"
    device = torch.device("cuda")

    weights = Path(args.weights) if args.weights else Path(run_dir) / "weights" / "best_model.pth"
    if not weights.exists():
        raise SystemExit(f"[中止] 未找到权重: {weights}")
    if not Path(run_dir).exists():
        raise SystemExit(f"[中止] 实验目录不存在: {run_dir}")

    tc = cfg["train"]
    root = cfg["data"]["root"]
    dataset = CamVidDataset(root, args.eval_split)
    loader = DataLoader(dataset, batch_size=int(tc["batch_size"]),
                        num_workers=int(tc["num_workers"]), pin_memory=True, shuffle=False)

    model = build_model().to(device)
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    best_epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    best_val = ckpt.get("val_miou") if isinstance(ckpt, dict) else None

    vis_samples = cfg.get("eval", {}).get("vis_samples", 12)
    evaluate_and_save(model, loader, dataset, device, run_dir, args.eval_split, exp_name,
                      seed, vis_samples, best_epoch, best_val)


# ---------------------------------------------------------------- 训练入口
def main():
    ap = argparse.ArgumentParser(description="CamVid E0 语义分割基准训练/评估")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖配置文件中的总 epoch 数")
    ap.add_argument("--resume", action="store_true", help="从 checkpoint_last.pth 继续训练")
    ap.add_argument("--eval-only", action="store_true", help="只加载 best_model.pth 评估，不训练")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟模式：写入 results/smoke/ 独立目录，不评估测试集")
    ap.add_argument("--run-dir", default=None, help="显式指定实验目录（eval-only/resume 常用）")
    ap.add_argument("--weights", default=None, help="eval-only 时指定权重，默认 <run-dir>/weights/best_model.pth")
    ap.add_argument("--eval-split", choices=["test", "val"], default="test",
                    help="eval-only 评估的 split，正式评估用 test")
    ap.add_argument("--skip-test", action="store_true",
                    help="训练结束后不评估测试集（工程调试/分批训练用，正式 E0 不要加）")
    args = ap.parse_args()
    started_at = time.time()

    if args.smoke and (args.resume or args.eval_only):
        raise SystemExit("--smoke 不能与 --resume / --eval-only 同用")

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    run_dir = resolve_run_dir(cfg, args)

    if args.eval_only:
        # 优先使用实验目录内的配置快照，保证模型结构与训练时一致
        snapshot = Path(run_dir) / "config.yaml"
        if snapshot.exists():
            cfg = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
        run_eval_only(cfg, run_dir, args)
        return

    exp_name = cfg.get("exp", {}).get("name", "E0")
    seed = int(cfg["seed"])
    tc = cfg["train"]
    if args.epochs is not None:
        epochs = args.epochs
    elif args.smoke:
        epochs = 1
        print("[smoke] 未指定 --epochs，默认 1 个 epoch")
    else:
        epochs = int(tc["epochs"])
    num_workers = int(tc["num_workers"])

    set_seed(seed)
    assert torch.cuda.is_available(), "需要 GPU"
    device = torch.device("cuda")

    # 目录与元数据
    if args.resume:
        ckpt_path = Path(run_dir) / "weights" / "checkpoint_last.pth"
        if not ckpt_path.exists():
            raise SystemExit(f"[中止] 未找到 checkpoint: {ckpt_path}")
        check_config_snapshot(run_dir, cfg)
    else:
        ensure_fresh_run_dir(run_dir)
        copy_config(args.config, run_dir)
        write_env(run_dir, seed, sys.argv, started_at)
    make_dirs(run_dir)
    log_path = Path(run_dir) / "metrics" / "train_log.csv"

    # 数据
    root, crop = cfg["data"]["root"], tuple(cfg["data"]["crop_size"])
    train_set = CamVidDataset(root, "train", crop)
    val_set = CamVidDataset(root, "val")
    kw = dict(batch_size=int(tc["batch_size"]), num_workers=num_workers, pin_memory=True)
    train_gen = torch.Generator()
    train_gen.manual_seed(seed)
    train_loader = DataLoader(
        train_set, shuffle=True, drop_last=True, generator=train_gen,
        worker_init_fn=_worker_init if num_workers > 0 else None, **kw)
    val_loader = DataLoader(val_set, shuffle=False, **kw)

    # 模型 / 损失 / 优化器（E0 定义保持不变）
    model = build_model().to(device)
    loss_fn = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tc["lr"]))
    scaler = torch.amp.GradScaler("cuda")

    if args.resume:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if int(ckpt["seed"]) != seed:
            raise SystemExit(f"[中止] seed 不一致: checkpoint={ckpt['seed']} config={seed}")
        done_epoch, best = restore_checkpoint(ckpt, model, optimizer, scaler, train_gen, device)
        start_epoch = done_epoch + 1
        truncate_log(log_path, done_epoch)
        print(f"[恢复] 从 epoch {start_epoch} 继续，已完成 {done_epoch}，best_val_miou={best:.4f}")
    else:
        write_log_header(log_path)
        start_epoch, best = 1, -1.0

    best_path = Path(run_dir) / "weights" / "best_model.pth"

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(normalize(x))
            loss = loss_fn(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        train_loss = running / len(train_loader)

        val = evaluate_split(model, val_loader, device, loss_fn)
        is_best = val["miou"] > best
        if is_best:
            best = val["miou"]
            atomic_torch_save({"model_state_dict": model.state_dict(),
                               "epoch": epoch, "val_miou": best,
                               "seed": seed, "exp": exp_name}, best_path)

        epoch_sec = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        append_log(log_path, [epoch, seed, f"{train_loss:.4f}", f"{val['loss']:.4f}",
                              f"{val['miou']:.4f}", f"{val['acc']:.4f}",
                              f"{lr_now:.2e}", f"{epoch_sec:.1f}"])
        save_checkpoint(Path(run_dir) / "weights" / "checkpoint_last.pth",
                        model, optimizer, scaler, epoch, best, seed, train_gen)
        print(f"epoch {epoch:02d}/{epochs} | train_loss {train_loss:.4f} | val_loss {val['loss']:.4f} "
              f"| val_mIoU {val['miou']:.4f} | {epoch_sec:.0f}s "
              f"| 峰值显存 {torch.cuda.max_memory_allocated() / 1e9:.2f}GB {'*best' if is_best else ''}")

    if args.smoke or args.skip_test:
        reason = "smoke" if args.smoke else "--skip-test"
        print(f"\n[{reason}] 跳过测试集评估；产物目录: {run_dir}")
        finish_env(run_dir, started_at,
                   {"mode": "smoke" if args.smoke else "skip_test",
                    "resumed": args.resume,
                    "epochs_done": epochs, "best_val_miou": best})
        return

    # 正式实验：训练结束、模型冻结后，测试集只评估一次
    test_metrics_path = Path(run_dir) / "metrics" / "test_metrics.csv"
    if test_metrics_path.exists():
        print("\n测试集正式评估产物已存在，跳过重复评估；如需重新生成请使用 --eval-only。")
    else:
        if not best_path.exists():
            raise SystemExit(f"[中止] 未找到 best_model.pth: {best_path}")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        test_set = CamVidDataset(root, "test")
        test_loader = DataLoader(test_set, shuffle=False, **kw)
        vis_samples = cfg.get("eval", {}).get("vis_samples", 12)
        evaluate_and_save(model, test_loader, test_set, device, run_dir, "test", exp_name,
                          seed, vis_samples, ckpt.get("epoch"), ckpt.get("val_miou"))

    finish_env(run_dir, started_at,
               {"mode": "formal", "resumed": args.resume,
                "epochs_done": epochs, "best_val_miou": best,
                "run_dir": str(run_dir)})


if __name__ == "__main__":      # Windows 下用 DataLoader 多进程时必须有这个保护
    main()
