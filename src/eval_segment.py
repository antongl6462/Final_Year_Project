#!/usr/bin/env python3
"""
Evaluate YOLO segmentation model for concrete crack detection.

Computes:
- Pixel-level Dice coefficient and IoU
- Union of instance masks into single binary mask
- Visual overlays of predictions vs ground truth
- Analysis of worst-performing cases

Usage:
    python src/eval_segment.py --model runs/segment/train/weights/best.pt --data_yaml /path/to/crack-seg.yaml
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO segmentation model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained YOLO model (.pt file)",
    )
    parser.add_argument(
        "--data_yaml",
        type=str,
        required=True,
        help="Path to YOLO dataset YAML file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate (default: test)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use (default: '' for auto-select)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=256,
        help="Image size (default: 256)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold for NMS (default: 0.7)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for evaluation results (default: model_dir/eval)",
    )
    parser.add_argument(
        "--save_worst",
        type=int,
        default=10,
        help="Number of worst cases to save (default: 10)",
    )
    parser.add_argument(
        "--save_best",
        type=int,
        default=5,
        help="Number of best cases to save (default: 5)",
    )
    return parser.parse_args()


def load_ground_truth_mask(label_path: Path, image_shape: Tuple[int, int]) -> np.ndarray:
    """Load YOLO polygon labels and convert to binary mask."""
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if not label_path.exists():
        return mask
    
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class_id + at least 3 points (x,y pairs)
                continue
            
            # Parse polygon coordinates (skip class_id)
            coords = list(map(float, parts[1:]))
            points = np.array(coords).reshape(-1, 2)
            
            # Denormalize coordinates
            points[:, 0] *= width
            points[:, 1] *= height
            points = points.astype(np.int32)
            
            # Fill polygon
            cv2.fillPoly(mask, [points], 255)
    
    return mask


def union_instance_masks(result) -> np.ndarray:
    """Union all instance masks from YOLO result into single binary mask."""
    if result.masks is None:
        # No masks predicted, return empty mask
        return np.zeros(result.orig_shape[:2], dtype=np.uint8)
    
    # Get all masks and union them
    masks = result.masks.data.cpu().numpy()  # Shape: (N, H, W)
    
    if len(masks) == 0:
        return np.zeros(result.orig_shape[:2], dtype=np.uint8)
    
    # Resize masks to original image size
    orig_h, orig_w = result.orig_shape[:2]
    binary_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    
    for mask in masks:
        # Resize mask to original size
        resized_mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        # Threshold at 0.5
        binary_mask = np.maximum(binary_mask, (resized_mask > 0.5).astype(np.uint8) * 255)
    
    return binary_mask


def compute_dice_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
    """Compute Dice coefficient and IoU for binary masks."""
    pred_binary = (pred_mask > 127).astype(np.uint8)
    gt_binary = (gt_mask > 127).astype(np.uint8)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    pred_sum = pred_binary.sum()
    gt_sum = gt_binary.sum()
    union = pred_sum + gt_sum - intersection
    
    # Dice coefficient
    if pred_sum + gt_sum == 0:
        dice = 1.0 if intersection == 0 else 0.0
    else:
        dice = 2.0 * intersection / (pred_sum + gt_sum)
    
    # IoU
    if union == 0:
        iou = 1.0 if intersection == 0 else 0.0
    else:
        iou = intersection / union
    
    return float(dice), float(iou)


def create_overlay_image(image: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """Create visualization overlay: GT in green, pred in red, overlap in yellow."""
    # Ensure image is RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Create overlay
    overlay = image.copy()
    
    pred_binary = (pred_mask > 127).astype(bool)
    gt_binary = (gt_mask > 127).astype(bool)
    
    # GT only (green)
    gt_only = np.logical_and(gt_binary, ~pred_binary)
    overlay[gt_only] = overlay[gt_only] * 0.5 + np.array([0, 255, 0]) * 0.5
    
    # Pred only (red)
    pred_only = np.logical_and(pred_binary, ~gt_binary)
    overlay[pred_only] = overlay[pred_only] * 0.5 + np.array([255, 0, 0]) * 0.5
    
    # Overlap (yellow)
    overlap = np.logical_and(pred_binary, gt_binary)
    overlay[overlap] = overlay[overlap] * 0.5 + np.array([255, 255, 0]) * 0.5
    
    return overlay.astype(np.uint8)


def evaluate_split(
    model: YOLO,
    data_root: Path,
    split: str,
    imgsz: int,
    conf: float,
    iou_thresh: float,
    device: str,
) -> Tuple[List[Dict], List[float], List[float]]:
    """Evaluate model on a data split."""
    images_dir = data_root / "images" / split
    labels_dir = data_root / "labels" / split
    
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    
    # Get all images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in image_extensions])
    
    print(f"Found {len(image_paths)} images in {split} set")
    
    results_list = []
    dice_scores = []
    iou_scores = []
    
    for image_path in tqdm(image_paths, desc=f"Evaluating {split}"):
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not read {image_path}")
            continue
        
        # Run inference
        results = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou_thresh,
            device=device,
            verbose=False,
        )
        
        if len(results) == 0:
            continue
        
        result = results[0]
        
        # Get predicted mask (union of all instances)
        pred_mask = union_instance_masks(result)
        
        # Load ground truth mask
        label_path = labels_dir / f"{image_path.stem}.txt"
        gt_mask = load_ground_truth_mask(label_path, image.shape[:2])
        
        # Compute metrics
        dice, iou = compute_dice_iou(pred_mask, gt_mask)
        
        dice_scores.append(dice)
        iou_scores.append(iou)
        
        results_list.append({
            "image_path": str(image_path),
            "image_name": image_path.name,
            "dice": dice,
            "iou": iou,
            "pred_mask": pred_mask,
            "gt_mask": gt_mask,
            "image": image,
        })
    
    return results_list, dice_scores, iou_scores


def save_visualization(
    results_list: List[Dict],
    output_dir: Path,
    prefix: str,
    num_samples: int,
    sort_key: str = "dice",
    ascending: bool = True,
):
    """Save visualization of best/worst cases."""
    # Sort results
    sorted_results = sorted(results_list, key=lambda x: x[sort_key], reverse=not ascending)
    selected = sorted_results[:num_samples]
    
    vis_dir = output_dir / f"{prefix}_cases"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, result in enumerate(selected):
        # Create overlay
        overlay = create_overlay_image(
            result["image"],
            result["pred_mask"],
            result["gt_mask"],
        )
        
        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(result["image"], cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Ground truth
        axes[1].imshow(result["gt_mask"], cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        
        # Prediction
        axes[2].imshow(result["pred_mask"], cmap="gray")
        axes[2].set_title("Prediction")
        axes[2].axis("off")
        
        # Overlay
        axes[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[3].set_title(f"Overlay (Dice={result['dice']:.4f}, IoU={result['iou']:.4f})")
        axes[3].axis("off")
        
        plt.suptitle(f"{prefix.capitalize()} #{idx+1}: {result['image_name']}", fontsize=14)
        plt.tight_layout()
        
        output_path = vis_dir / f"{prefix}_{idx+1:03d}_{result['image_name']}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    
    print(f"✓ Saved {len(selected)} {prefix} cases to {vis_dir}")


def main():
    args = parse_args()
    
    # Resolve paths
    model_path = Path(args.model).resolve()
    data_yaml = Path(args.data_yaml).resolve()
    
    if not model_path.exists():
        raise ValueError(f"Model not found: {model_path}")
    
    if not data_yaml.exists():
        raise ValueError(f"Data YAML not found: {data_yaml}")
    
    # Parse data root from YAML
    import yaml
    with open(data_yaml, "r") as f:
        data_config = yaml.safe_load(f)
    
    data_root = Path(data_config["path"])
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = model_path.parent.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("YOLO SEGMENTATION EVALUATION")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Data YAML: {data_yaml}")
    print(f"Data root: {data_root}")
    print(f"Split: {args.split}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device if args.device else 'auto'}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print()
    
    # Load model
    print("Loading model...")
    model = YOLO(str(model_path))
    print("✓ Model loaded")
    print()
    
    # Evaluate
    print(f"Evaluating on {args.split} set...")
    results_list, dice_scores, iou_scores = evaluate_split(
        model,
        data_root,
        args.split,
        args.imgsz,
        args.conf,
        args.iou,
        args.device,
    )
    
    print("✓ Evaluation complete")
    print()
    
    # Compute statistics
    mean_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    mean_iou = np.mean(iou_scores)
    std_iou = np.std(iou_scores)
    
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Number of images: {len(results_list)}")
    print()
    print(f"Dice Coefficient:")
    print(f"  Mean: {mean_dice:.4f}")
    print(f"  Std:  {std_dice:.4f}")
    print(f"  Min:  {np.min(dice_scores):.4f}")
    print(f"  Max:  {np.max(dice_scores):.4f}")
    print()
    print(f"IoU (Intersection over Union):")
    print(f"  Mean: {mean_iou:.4f}")
    print(f"  Std:  {std_iou:.4f}")
    print(f"  Min:  {np.min(iou_scores):.4f}")
    print(f"  Max:  {np.max(iou_scores):.4f}")
    print()
    
    # Save metrics
    metrics = {
        "split": args.split,
        "num_images": len(results_list),
        "dice": {
            "mean": float(mean_dice),
            "std": float(std_dice),
            "min": float(np.min(dice_scores)),
            "max": float(np.max(dice_scores)),
        },
        "iou": {
            "mean": float(mean_iou),
            "std": float(std_iou),
            "min": float(np.min(iou_scores)),
            "max": float(np.max(iou_scores)),
        },
        "per_image": [
            {
                "image_name": r["image_name"],
                "dice": r["dice"],
                "iou": r["iou"],
            }
            for r in results_list
        ],
    }
    
    metrics_file = output_dir / f"metrics_{args.split}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_file}")
    
    # Save visualizations
    if args.save_worst > 0:
        print(f"\nSaving {args.save_worst} worst cases...")
        save_visualization(
            results_list,
            output_dir,
            "worst",
            args.save_worst,
            sort_key="dice",
            ascending=True,
        )
    
    if args.save_best > 0:
        print(f"\nSaving {args.save_best} best cases...")
        save_visualization(
            results_list,
            output_dir,
            "best",
            args.save_best,
            sort_key="dice",
            ascending=False,
        )
    
    # Plot distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(dice_scores, bins=30, edgecolor="black", alpha=0.7)
    axes[0].axvline(mean_dice, color="red", linestyle="--", linewidth=2, label=f"Mean={mean_dice:.4f}")
    axes[0].set_xlabel("Dice Coefficient", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("Dice Coefficient Distribution", fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].hist(iou_scores, bins=30, edgecolor="black", alpha=0.7)
    axes[1].axvline(mean_iou, color="red", linestyle="--", linewidth=2, label=f"Mean={mean_iou:.4f}")
    axes[1].set_xlabel("IoU", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("IoU Distribution", fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    dist_plot = output_dir / f"metrics_distribution_{args.split}.png"
    plt.savefig(dist_plot, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Distribution plot saved to: {dist_plot}")
    
    print()
    print("=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
