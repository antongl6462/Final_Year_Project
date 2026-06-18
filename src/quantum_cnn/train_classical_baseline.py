"""Training script for ClassicalCNNBaseline.

Reuses the EXACT same data pipeline, split, seeds, and hyperparameters as
the QCNN notebook so metric differences isolate the quantum contribution.

Usage (from project root, with .venv active):
    python -m src.quantum_cnn.train_classical_baseline

Optional flags:
    --subset 40000   # NOT like-for-like with QCNN; prints a warning
    --epochs 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — allow running as a script from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantum_cnn.classical_baseline import ClassicalCNNBaseline, count_parameters, print_param_comparison
from quantum_cnn.config import QuantumCNNConfig
from quantum_cnn.data import create_dataloaders
from quantum_cnn.hybrid_qcnn import HybridQuantumCNN
from quantum_cnn.train_quantum_cnn import evaluate_model, train_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Plotting helpers (same style as run_experiment.py / QCNN notebook)
# ---------------------------------------------------------------------------
def plot_curves(history: dict, output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 4), facecolor="white")

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs, history["val_loss"],   marker="o", label="Val Loss",   color="#ff7f0e", linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curves")
    plt.legend(); plt.grid(alpha=0.3)
    plt.gca().set_facecolor("white")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], marker="o", label="Train Acc", color="#1f77b4", linewidth=2)
    plt.plot(epochs, history["val_acc"],   marker="o", label="Val Acc",   color="#ff7f0e", linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy Curves")
    plt.legend(); plt.grid(alpha=0.3)
    plt.gca().set_facecolor("white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm_values: list, class_names: list, output_path: Path) -> None:
    matrix = np.array(cm_values)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Classical CNN Baseline — Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(subset_size: int = 10_000, epochs: int = 8) -> None:
    # ------------------------------------------------------------------
    # Warn if subset is not like-for-like with QCNN
    # ------------------------------------------------------------------
    QCNN_SUBSET = 10_000
    if subset_size != QCNN_SUBSET:
        print(
            f"\n⚠️  WARNING: subset_size={subset_size:,} ≠ {QCNN_SUBSET:,} (QCNN subset).\n"
            "   This run is NOT like-for-like with the QCNN. Results are NOT directly comparable.\n"
        )

    SEED = 42
    set_all_seeds(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Seed   : {SEED}")

    # ------------------------------------------------------------------
    # Config — EXACTLY matching the QCNN notebook cell values
    # ------------------------------------------------------------------
    max_train = int(subset_size * 0.70)
    max_val   = int(subset_size * 0.15)
    max_test  = int(subset_size * 0.15)

    config = QuantumCNNConfig(
        dataset_root=PROJECT_ROOT / "raw_classification",
        image_size=32,
        batch_size=32,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=epochs,
        seed=SEED,
        n_qubits=4,
        n_q_layers=2,
        quantum_embedding_dim=8,
        hidden_dim=64,
        max_train_samples=max_train,
        max_val_samples=max_val,
        max_test_samples=max_test,
        output_root=PROJECT_ROOT / "runs" / "quantum_cnn",
        experiment_name="notebook_run",
    )

    # Output directory (separate from QCNN)
    out_dir = PROJECT_ROOT / "runs" / "classical_baseline" / "notebook_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data — same pipeline, same seeds → identical split to QCNN
    # ------------------------------------------------------------------
    train_loader, val_loader, test_loader, classes = create_dataloaders(config)
    print(f"\nClasses : {classes}")
    print(f"Train batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")
    print(f"Test  batches : {len(test_loader)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = ClassicalCNNBaseline(
        num_classes=len(classes),
        quantum_embedding_dim=config.quantum_embedding_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)

    # Parameter comparison (requires a throwaway QCNN instance)
    qcnn_ref = HybridQuantumCNN(
        num_classes=len(classes),
        n_qubits=config.n_qubits,
        n_q_layers=config.n_q_layers,
        quantum_embedding_dim=config.quantum_embedding_dim,
        hidden_dim=config.hidden_dim,
    )
    print("\n--- Trainable Parameter Comparison ---")
    print_param_comparison(qcnn_ref, model)
    del qcnn_ref  # free memory

    # ------------------------------------------------------------------
    # Train — reuses train_quantum_cnn.train_model verbatim
    # NOTE: train_model uses torch.optim.Adam (not AdamW) — we match
    # this exactly so the comparison is like-for-like.
    # ------------------------------------------------------------------
    set_all_seeds(SEED)  # re-seed before training
    t0 = time.time()
    history, best_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=device,
    )
    train_time = time.time() - t0
    print(f"\nTraining time : {train_time / 60:.2f} min  ({train_time:.1f} s)")
    print(f"Device used   : {device}")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    metrics = evaluate_model(best_model, test_loader, classes, device)

    print(f"\nTest Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision     : {metrics['precision']:.4f}")
    print(f"Recall        : {metrics['recall']:.4f}")
    print(f"F1            : {metrics['f1']:.4f}")
    print(f"ROC-AUC       : {metrics['roc_auc']:.4f}")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    # Model checkpoint
    model_path = out_dir / "best_classical_baseline.pt"
    torch.save(best_model.state_dict(), model_path)

    # Metrics JSON
    metrics_payload = {
        "model": "ClassicalCNNBaseline",
        "subset_size": subset_size,
        "epochs": epochs,
        "device": device,
        "train_time_seconds": round(train_time, 2),
        "test_accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "confusion_matrix": metrics["confusion_matrix"],
        "history": history,
        "config": config.to_dict(),
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics_payload, fh, indent=2, default=str)

    # Plots
    plot_curves(history, out_dir / "training_curves.png")
    plot_confusion_matrix(metrics["confusion_matrix"], classes, out_dir / "confusion_matrix.png")

    print(f"\nSaved model   : {model_path}")
    print(f"Saved metrics : {metrics_path}")
    print(f"Saved plots   : {out_dir}")

    # ------------------------------------------------------------------
    # Comparison summary (loads QCNN metrics if available)
    # ------------------------------------------------------------------
    qcnn_metrics_path = PROJECT_ROOT / "runs" / "quantum_cnn" / "notebook_run" / "metrics.json"
    _print_comparison_table(
        baseline_metrics=metrics_payload,
        qcnn_path=qcnn_metrics_path,
        train_time=train_time,
        device=device,
        baseline_params=count_parameters(best_model),
    )

    return metrics_payload


def _print_comparison_table(
    baseline_metrics: dict,
    qcnn_path: Path,
    train_time: float,
    device: str,
    baseline_params: int,
) -> None:
    qcnn = {}
    if qcnn_path.exists():
        with open(qcnn_path) as fh:
            qcnn = json.load(fh)

    qcnn_params_str = f"{qcnn.get('total_params', 'N/A')}"

    print("\n" + "=" * 75)
    print("  COMPARISON SUMMARY: QCNN  vs  Classical CNN Baseline")
    print("=" * 75)
    fmt = "{:<28} {:>20} {:>20}"
    print(fmt.format("Metric", "QCNN", "Classical Baseline"))
    print("-" * 75)
    print(fmt.format("Total parameters",
                      qcnn_params_str,
                      f"{baseline_params:,}"))
    print(fmt.format("Test Accuracy",
                      f"{qcnn.get('test_accuracy', 'N/A')}",
                      f"{baseline_metrics['test_accuracy']:.4f}"))
    print(fmt.format("Precision",
                      f"{qcnn.get('precision', 'N/A')}",
                      f"{baseline_metrics['precision']:.4f}"))
    print(fmt.format("Recall",
                      f"{qcnn.get('recall', 'N/A')}",
                      f"{baseline_metrics['recall']:.4f}"))
    print(fmt.format("F1 Score",
                      f"{qcnn.get('f1', 'N/A')}",
                      f"{baseline_metrics['f1']:.4f}"))
    print(fmt.format("ROC-AUC",
                      f"{qcnn.get('roc_auc', 'N/A')}",
                      f"{baseline_metrics['roc_auc']:.4f}"))
    print(fmt.format("Training Time",
                      f"{qcnn.get('train_time_seconds', 'N/A')}",
                      f"{train_time:.1f} s"))
    print(fmt.format("Device",
                      f"{qcnn.get('device', 'N/A')}",
                      device))
    print("=" * 75)
    if not qcnn_path.exists():
        print("  ℹ️  QCNN metrics.json not found — run the QCNN notebook first.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Classical CNN Baseline")
    parser.add_argument("--subset", type=int, default=10_000,
                        help="Total subset size (default 10000 = like-for-like with QCNN)")
    parser.add_argument("--epochs", type=int, default=8,
                        help="Number of training epochs (default 8)")
    args = parser.parse_args()
    main(subset_size=args.subset, epochs=args.epochs)
