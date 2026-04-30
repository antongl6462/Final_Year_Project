from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from skimage.morphology import skeletonize
except Exception:
    skeletonize = None


EPS = 1e-7


@dataclass
class PostProcessingConfig:
    min_component_area: int = 0
    closing_kernel: int = 0
    dilation_kernel: int = 0


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_term = (1.0 - p_t).pow(self.gamma)
        return (alpha_t * focal_term * bce).mean()


class CombinedSegmentationLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float,
        bce_weight: float = 0.4,
        dice_weight: float = 0.4,
        focal_weight: float = 0.2,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.register_buffer("pos_weight_tensor", torch.tensor([pos_weight], dtype=torch.float32))
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_tensor)
        self.dice = DiceLoss()
        self.focal = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.bce_weight * self.bce(logits, targets)
            + self.dice_weight * self.dice(logits, targets)
            + self.focal_weight * self.focal(logits, targets)
        )


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).float()
    pred = pred.view(pred.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (pred * targets).sum(dim=1)
    denom = pred.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + 1.0) / (denom + 1.0)
    return float(dice.mean().item())


def apply_post_processing(mask: np.ndarray, config: Optional[PostProcessingConfig]) -> np.ndarray:
    if config is None:
        return mask.astype(np.uint8)

    result = mask.astype(np.uint8).copy()
    if config.closing_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.closing_kernel, config.closing_kernel))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    if config.dilation_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.dilation_kernel, config.dilation_kernel))
        result = cv2.dilate(result, kernel, iterations=1)
    if config.min_component_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
        filtered = np.zeros_like(result)
        for label_index in range(1, num_labels):
            if stats[label_index, cv2.CC_STAT_AREA] >= config.min_component_area:
                filtered[labels == label_index] = 1
        result = filtered
    return result.astype(np.uint8)


def confusion_from_masks(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[int, int, int, int]:
    pred = (pred_mask > 0).astype(np.uint8)
    target = (gt_mask > 0).astype(np.uint8)
    tp = int(np.logical_and(pred == 1, target == 1).sum())
    fp = int(np.logical_and(pred == 1, target == 0).sum())
    tn = int(np.logical_and(pred == 0, target == 0).sum())
    fn = int(np.logical_and(pred == 0, target == 1).sum())
    return tp, fp, tn, fn


def compute_binary_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    tp, fp, tn, fn = confusion_from_masks(pred_mask, gt_mask)
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    specificity = tn / (tn + fp + EPS)
    fpr = fp / (fp + tn + EPS)
    fnr = fn / (fn + tp + EPS)
    iou = tp / (tp + fp + fn + EPS)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + EPS)
    accuracy = (tp + tn) / (tp + fp + tn + fn + EPS)
    f1 = 2.0 * precision * recall / (precision + recall + EPS)
    metrics = {
        "pixel_accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "dice": float(dice),
        "iou": float(iou),
        "specificity": float(specificity),
        "false_negative_rate": float(fnr),
        "false_positive_rate": float(fpr),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }
    if skeletonize is not None:
        pred_skeleton = skeletonize(pred_mask > 0)
        gt_skeleton = skeletonize(gt_mask > 0)
        intersection = np.logical_and(pred_skeleton, gt_skeleton).sum()
        union = np.logical_or(pred_skeleton, gt_skeleton).sum()
        metrics["skeleton_iou"] = float(intersection / (union + EPS))
    return metrics


def aggregate_confusion_metrics(metric_rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    tp = sum(int(row.get("tp", 0.0)) for row in metric_rows)
    fp = sum(int(row.get("fp", 0.0)) for row in metric_rows)
    tn = sum(int(row.get("tn", 0.0)) for row in metric_rows)
    fn = sum(int(row.get("fn", 0.0)) for row in metric_rows)
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    specificity = tn / (tn + fp + EPS)
    fpr = fp / (fp + tn + EPS)
    fnr = fn / (fn + tp + EPS)
    iou = tp / (tp + fp + fn + EPS)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + EPS)
    accuracy = (tp + tn) / (tp + fp + tn + fn + EPS)
    f1 = 2.0 * precision * recall / (precision + recall + EPS)
    result = {
        "pixel_accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "dice": float(dice),
        "iou": float(iou),
        "specificity": float(specificity),
        "false_negative_rate": float(fnr),
        "false_positive_rate": float(fpr),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "mean_per_image_dice": float(np.mean([row["dice"] for row in metric_rows])) if metric_rows else 0.0,
        "mean_per_image_iou": float(np.mean([row["iou"] for row in metric_rows])) if metric_rows else 0.0,
    }
    if metric_rows and "skeleton_iou" in metric_rows[0]:
        result["mean_skeleton_iou"] = float(np.mean([row.get("skeleton_iou", 0.0) for row in metric_rows]))
    return result


def threshold_sweep(
    probability_maps: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    thresholds: Iterable[float],
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for threshold in thresholds:
        metric_rows = []
        for probability_map, gt_mask in zip(probability_maps, gt_masks):
            pred_mask = (probability_map >= threshold).astype(np.uint8)
            metric_rows.append(compute_binary_metrics(pred_mask, gt_mask))
        summary = aggregate_confusion_metrics(metric_rows)
        summary["threshold"] = float(threshold)
        rows.append(summary)
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def select_operating_threshold(threshold_df: pd.DataFrame, recall_target: float = 0.85) -> Tuple[float, float]:
    best_f1_row = threshold_df.sort_values(["f1", "recall"], ascending=[False, False]).iloc[0]
    recall_candidates = threshold_df[threshold_df["recall"] >= recall_target]
    if len(recall_candidates) > 0:
        recall_row = recall_candidates.sort_values(["f1", "recall"], ascending=[False, False]).iloc[0]
    else:
        recall_row = threshold_df.sort_values(["recall", "f1"], ascending=[False, False]).iloc[0]
    return float(best_f1_row["threshold"]), float(recall_row["threshold"])
