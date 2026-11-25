import torch
from dataset import NUM_CLASSES, VOID


def confusion_matrix(pred, target, num_classes=NUM_CLASSES, ignore_index=VOID):
    """pred/target: (B,H,W) 整数张量。返回 (C,C)：行=真实类，列=预测类。void 像素被排除。"""
    mask = target != ignore_index
    idx = target[mask] * num_classes + pred[mask]
    return torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def iou_from_cm(cm):
    """每类 IoU。某类在真值和预测里都没出现时为 nan（0/0），不参与平均。"""
    cm = cm.double()
    inter = cm.diag()
    union = cm.sum(0) + cm.sum(1) - inter
    return inter / union