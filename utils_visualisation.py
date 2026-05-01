from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_mask_sanity_grid(samples: Sequence[Dict], output_path: Path | None = None) -> None:
    if not samples:
        return
    fig, axes = plt.subplots(len(samples), 4, figsize=(18, 4 * len(samples)))
    if len(samples) == 1:
        axes = np.array(axes).reshape(1, 4)
    for row, sample in enumerate(samples):
        axes[row, 0].imshow(sample["image"])
        axes[row, 0].set_title(f"Image\n{sample['stem']}")
        axes[row, 1].imshow(sample["raw_mask"], cmap="gray")
        axes[row, 1].set_title("Raw Mask")
        axes[row, 2].imshow(sample["binary_mask"], cmap="gray")
        axes[row, 2].set_title("Binary Mask")
        axes[row, 3].imshow(sample["overlay"])
        axes[row, 3].set_title("Mask Overlay")
        for col in range(4):
            axes[row, col].axis("off")
    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(history_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    curves = [
        ("train_loss", "val_loss", "Loss", axes[0, 0]),
        ("val_dice", None, "Validation Dice", axes[0, 1]),
        ("val_iou", None, "Validation IoU", axes[0, 2]),
        ("val_precision", None, "Validation Precision", axes[1, 0]),
        ("val_recall", None, "Validation Recall", axes[1, 1]),
        ("val_f1", None, "Validation F1", axes[1, 2]),
    ]
    for primary, secondary, title, axis in curves:
        if primary in history_df:
            axis.plot(history_df["epoch"], history_df[primary], label=primary)
        if secondary and secondary in history_df:
            axis.plot(history_df["epoch"], history_df[secondary], label=secondary)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
        if axis.get_legend_handles_labels()[0]:
            axis.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_threshold_curves(threshold_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for column in ["precision", "recall", "f1", "dice", "iou"]:
        if column in threshold_df:
            ax.plot(threshold_df["threshold"], threshold_df[column], marker="o", label=column)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.set_title("Threshold Calibration Curves")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(confusion: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Background", "Crack"])
    ax.set_yticklabels(["Background", "Crack"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Pixel-Level Confusion Matrix")
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            ax.text(col, row, f"{int(confusion[row, col])}", ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _error_map(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    canvas = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    tp = np.logical_and(pred_mask == 1, gt_mask == 1)
    fp = np.logical_and(pred_mask == 1, gt_mask == 0)
    fn = np.logical_and(pred_mask == 0, gt_mask == 1)
    canvas[tp] = (255, 255, 255)
    canvas[fp] = (255, 0, 0)
    canvas[fn] = (0, 255, 0)
    return canvas


def save_qualitative_predictions(items: Sequence[Dict], output_path: Path, title: str, max_items: int = 6) -> None:
    if not items:
        return
    selected = list(items[:max_items])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(selected), 5, figsize=(20, 4 * len(selected)))
    if len(selected) == 1:
        axes = np.array(axes).reshape(1, 5)
    for row, item in enumerate(selected):
        axes[row, 0].imshow(item["image"])
        axes[row, 0].set_title(f"Input\n{item['stem']}")
        axes[row, 1].imshow(item["gt_mask"], cmap="gray")
        axes[row, 1].set_title("Ground Truth")
        axes[row, 2].imshow(item["probability_map"], cmap="magma", vmin=0.0, vmax=1.0)
        axes[row, 2].set_title("Probability Heatmap")
        axes[row, 3].imshow(item["pred_mask"], cmap="gray")
        axes[row, 3].set_title(f"Prediction\nDice={item['dice']:.3f}")
        axes[row, 4].imshow(_error_map(item["pred_mask"], item["gt_mask"]))
        axes[row, 4].set_title("Error Map\nWhite=TP, Red=FP, Green=FN")
        for col in range(5):
            axes[row, col].axis("off")
    plt.suptitle(title, fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_worst_predictions(items: Sequence[Dict], output_path: Path, max_items: int = 6) -> None:
    ranked = sorted(items, key=lambda item: item.get("dice", 0.0))
    save_qualitative_predictions(ranked[:max_items], output_path, "Worst Mask Segmentation Predictions", max_items=max_items)
