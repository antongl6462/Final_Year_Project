"""Training and evaluation helpers for the hybrid QCNN experiment."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def _run_epoch(model, loader, criterion, optimizer, device: str, train_mode: bool) -> Tuple[float, float]:
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(train_mode):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
):
    """Train hybrid model and return history plus best checkpoint state dict."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None

    for epoch in range(epochs):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device, train_mode=True)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer, device, train_mode=False)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history, model


def evaluate_model(model, test_loader, classes, device: str) -> Dict:
    """Evaluate model and return dissertation-friendly metrics."""
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_probs.extend(probs.cpu().tolist())

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    y_probs_np = np.array(y_probs)

    metrics = {
        "accuracy": float((y_true_np == y_pred_np).mean()),
        "precision": float(classification_report(y_true_np, y_pred_np, target_names=classes, output_dict=True)[classes[1]]["precision"]) if len(classes) == 2 else None,
        "recall": float(classification_report(y_true_np, y_pred_np, target_names=classes, output_dict=True)[classes[1]]["recall"]) if len(classes) == 2 else None,
        "f1": float(classification_report(y_true_np, y_pred_np, target_names=classes, output_dict=True)[classes[1]]["f1-score"]) if len(classes) == 2 else None,
        "classification_report": classification_report(
            y_true_np,
            y_pred_np,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true_np, y_pred_np).tolist(),
    }

    if len(classes) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true_np, y_probs_np[:, 1]))

    return metrics
