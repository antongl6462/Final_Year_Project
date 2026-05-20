"""End-to-end experiment runner for the hybrid QCNN branch."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from .config import QuantumCNNConfig
from .data import create_dataloaders
from .hybrid_qcnn import HybridQuantumCNN
from .train_quantum_cnn import evaluate_model, train_model


def _plot_curves(history, output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Val Loss")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], marker="o", label="Train Acc")
    plt.plot(epochs, history["val_acc"], marker="o", label="Val Acc")
    plt.title("Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()


def _plot_confusion_matrix(confusion_matrix_values, class_names, output_path: Path) -> None:
    matrix = np.array(confusion_matrix_values)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Hybrid QCNN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()


def run_experiment(config: QuantumCNNConfig):
    """Run full hybrid QCNN experiment and save all outputs."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = config.output_root / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, classes = create_dataloaders(config)

    model = HybridQuantumCNN(
        num_classes=len(classes),
        n_qubits=config.n_qubits,
        n_q_layers=config.n_q_layers,
        quantum_embedding_dim=config.quantum_embedding_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)

    start_time = time.time()
    history, best_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=device,
    )
    elapsed = time.time() - start_time

    metrics = evaluate_model(best_model, test_loader, classes, device=device)

    model_path = output_dir / "best_hybrid_quantum_cnn.pt"
    torch.save(best_model.state_dict(), model_path)

    history_path = output_dir / "history.json"
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    _plot_curves(history, output_dir / "training_curves.png")
    _plot_confusion_matrix(metrics["confusion_matrix"], classes, output_dir / "confusion_matrix.png")

    config.save_json(output_dir / "config.json")

    summary_path = output_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("# Hybrid QCNN Experiment Summary\n\n")
        handle.write("## Scientific Positioning\n")
        handle.write("This is a proof-of-concept hybrid quantum-classical simulation; no claim of quantum advantage is made.\n\n")
        handle.write("## Main Results\n")
        handle.write(f"- Accuracy: {metrics['accuracy']:.4f}\n")
        if metrics.get("precision") is not None:
            handle.write(f"- Precision: {metrics['precision']:.4f}\n")
        if metrics.get("recall") is not None:
            handle.write(f"- Recall: {metrics['recall']:.4f}\n")
        if metrics.get("f1") is not None:
            handle.write(f"- F1-score: {metrics['f1']:.4f}\n")
        if "roc_auc" in metrics:
            handle.write(f"- ROC-AUC: {metrics['roc_auc']:.4f}\n")
        handle.write(f"- Training time (s): {elapsed:.2f}\n")

    print(f"Classes: {classes}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Artifacts directory: {output_dir}")
    return {
        "metrics": metrics,
        "history": history,
        "output_dir": str(output_dir),
    }
