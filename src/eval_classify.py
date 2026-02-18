#!/usr/bin/env python3
"""
Evaluate YOLO classification model for concrete crack detection.

Computes:
- Precision, Recall, F1-score
- ROC-AUC
- Confusion matrix
- Per-class metrics
- Optional threshold sweep on validation set

Usage:
    python src/eval_classify.py --model runs/classify/train/weights/best.pt --data_root /path/to/data
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO classification model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained YOLO model (.pt file)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing classify_yolo/ folder",
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
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Image size (default: 224)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for evaluation results (default: model_dir/eval)",
    )
    parser.add_argument(
        "--threshold_sweep",
        action="store_true",
        help="Perform threshold sweep on validation set",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save predictions to JSON file",
    )
    return parser.parse_args()


def get_image_paths_and_labels(data_dir: Path, split: str) -> Tuple[List[Path], List[str], List[int]]:
    """Get image paths, class names, and labels from YOLO classification structure."""
    split_dir = data_dir / split
    
    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")
    
    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class directories found in {split_dir}")
    
    class_names = [d.name for d in class_dirs]
    image_paths = []
    labels = []
    
    for label_idx, class_dir in enumerate(class_dirs):
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        images = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in extensions])
        image_paths.extend(images)
        labels.extend([label_idx] * len(images))
    
    return image_paths, class_names, labels


def predict_batch(model: YOLO, image_paths: List[Path], batch_size: int, imgsz: int, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on a batch of images."""
    all_probs = []
    all_preds = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Inference"):
        batch_paths = image_paths[i : i + batch_size]
        
        # Run inference
        results = model.predict(
            source=[str(p) for p in batch_paths],
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        
        # Extract predictions
        for result in results:
            probs = result.probs.data.cpu().numpy()  # Probability distribution
            pred_class = result.probs.top1  # Predicted class index
            all_probs.append(probs)
            all_preds.append(pred_class)
    
    return np.array(all_probs), np.array(all_preds)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray, class_names: List[str]) -> Dict:
    """Compute classification metrics."""
    metrics = {}
    
    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics["classification_report"] = report
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    
    # ROC-AUC (for binary classification)
    if len(class_names) == 2:
        # Use probability of positive class
        y_score = y_probs[:, 1] if y_probs.shape[1] == 2 else y_probs[:, 0]
        roc_auc = roc_auc_score(y_true, y_score)
        metrics["roc_auc"] = float(roc_auc)
        
        # ROC curve data
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        metrics["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }
        
        # Precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        metrics["pr_curve"] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        }
    
    # Overall accuracy
    accuracy = np.mean(y_true == y_pred)
    metrics["accuracy"] = float(accuracy)
    
    return metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: Path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Count"},
    )
    
    plt.title("Confusion Matrix (Normalized)", fontsize=14, pad=20)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Confusion matrix saved to: {output_path}")


def plot_roc_curve(metrics: Dict, output_path: Path):
    """Plot and save ROC curve."""
    if "roc_curve" not in metrics:
        return
    
    fpr = metrics["roc_curve"]["fpr"]
    tpr = metrics["roc_curve"]["tpr"]
    roc_auc = metrics["roc_auc"]
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ ROC curve saved to: {output_path}")


def plot_pr_curve(metrics: Dict, output_path: Path):
    """Plot and save precision-recall curve."""
    if "pr_curve" not in metrics:
        return
    
    precision = metrics["pr_curve"]["precision"]
    recall = metrics["pr_curve"]["recall"]
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color="darkorange", lw=2, label="Precision-Recall curve")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve", fontsize=14, pad=20)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Precision-Recall curve saved to: {output_path}")


def threshold_sweep_binary(y_true: np.ndarray, y_probs: np.ndarray, output_path: Path):
    """Perform threshold sweep for binary classification."""
    # Use probability of positive class
    y_score = y_probs[:, 1] if y_probs.shape[1] == 2 else y_probs[:, 0]
    
    thresholds = np.linspace(0, 1, 101)
    metrics_list = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_score >= threshold).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred_thresh == 1))
        tn = np.sum((y_true == 0) & (y_pred_thresh == 0))
        fp = np.sum((y_true == 0) & (y_pred_thresh == 1))
        fn = np.sum((y_true == 1) & (y_pred_thresh == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(y_true)
        
        metrics_list.append({
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
        })
    
    # Find best threshold by F1
    best_metric = max(metrics_list, key=lambda x: x["f1"])
    
    # Save results
    results = {
        "best_threshold": best_metric,
        "all_thresholds": metrics_list,
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Threshold sweep results saved to: {output_path}")
    print(f"  Best threshold: {best_metric['threshold']:.4f} (F1={best_metric['f1']:.4f})")
    
    # Plot threshold sweep
    plot_path = output_path.parent / "threshold_sweep.png"
    fig, ax = plt.subplots(figsize=(12, 6))
    
    thresholds_array = [m["threshold"] for m in metrics_list]
    ax.plot(thresholds_array, [m["precision"] for m in metrics_list], label="Precision", linewidth=2)
    ax.plot(thresholds_array, [m["recall"] for m in metrics_list], label="Recall", linewidth=2)
    ax.plot(thresholds_array, [m["f1"] for m in metrics_list], label="F1-Score", linewidth=2)
    ax.axvline(best_metric["threshold"], color="red", linestyle="--", label=f"Best threshold ({best_metric['threshold']:.2f})")
    
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Threshold Sweep Analysis", fontsize=14, pad=20)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Threshold sweep plot saved to: {plot_path}")


def main():
    args = parse_args()
    
    # Resolve paths
    model_path = Path(args.model).resolve()
    data_root = Path(args.data_root).resolve()
    classify_dir = data_root / "classify_yolo"
    
    if not model_path.exists():
        raise ValueError(f"Model not found: {model_path}")
    
    if not classify_dir.exists():
        raise ValueError(f"YOLO classification directory not found: {classify_dir}")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = model_path.parent.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("YOLO CLASSIFICATION EVALUATION")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Data root: {data_root}")
    print(f"Split: {args.split}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device if args.device else 'auto'}")
    print()
    
    # Load model
    print("Loading model...")
    model = YOLO(str(model_path))
    print("✓ Model loaded")
    print()
    
    # Get data
    print(f"Loading {args.split} data...")
    image_paths, class_names, y_true = get_image_paths_and_labels(classify_dir, args.split)
    y_true = np.array(y_true)
    
    print(f"✓ Loaded {len(image_paths)} images")
    print(f"  Classes: {class_names}")
    print(f"  Distribution: {np.bincount(y_true)}")
    print()
    
    # Run inference
    print("Running inference...")
    y_probs, y_pred = predict_batch(model, image_paths, args.batch, args.imgsz, args.device)
    print("✓ Inference complete")
    print()
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(y_true, y_pred, y_probs, class_names)
    print("✓ Metrics computed")
    print()
    
    # Print results
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    if "roc_auc" in metrics:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print()
    
    print("Per-class metrics:")
    for class_name in class_names:
        if class_name in metrics["classification_report"]:
            class_metrics = metrics["classification_report"][class_name]
            print(f"  {class_name}:")
            print(f"    Precision: {class_metrics['precision']:.4f}")
            print(f"    Recall: {class_metrics['recall']:.4f}")
            print(f"    F1-score: {class_metrics['f1-score']:.4f}")
            print(f"    Support: {class_metrics['support']}")
    print()
    
    # Save metrics
    metrics_file = output_dir / f"metrics_{args.split}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_file}")
    
    # Plot confusion matrix
    cm = np.array(metrics["confusion_matrix"])
    cm_file = output_dir / f"confusion_matrix_{args.split}.png"
    plot_confusion_matrix(cm, class_names, cm_file)
    
    # Plot ROC curve
    if "roc_curve" in metrics:
        roc_file = output_dir / f"roc_curve_{args.split}.png"
        plot_roc_curve(metrics, roc_file)
    
    # Plot PR curve
    if "pr_curve" in metrics:
        pr_file = output_dir / f"pr_curve_{args.split}.png"
        plot_pr_curve(metrics, pr_file)
    
    # Threshold sweep
    if args.threshold_sweep and len(class_names) == 2:
        print("\nPerforming threshold sweep...")
        sweep_file = output_dir / f"threshold_sweep_{args.split}.json"
        threshold_sweep_binary(y_true, y_probs, sweep_file)
    
    # Save predictions
    if args.save_predictions:
        predictions_file = output_dir / f"predictions_{args.split}.json"
        predictions = {
            "image_paths": [str(p) for p in image_paths],
            "true_labels": y_true.tolist(),
            "predicted_labels": y_pred.tolist(),
            "probabilities": y_probs.tolist(),
            "class_names": class_names,
        }
        with open(predictions_file, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"✓ Predictions saved to: {predictions_file}")
    
    print()
    print("=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
