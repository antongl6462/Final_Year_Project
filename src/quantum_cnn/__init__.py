"""Hybrid QCNN experiment package.

This package is intentionally separated from the existing YOLO pipeline so the
classical baseline remains unchanged and directly reusable.
"""

from .config import QuantumCNNConfig
from .data import create_dataloaders
from .hybrid_qcnn import HybridQuantumCNN
from .run_experiment import run_experiment

__all__ = [
    "QuantumCNNConfig",
    "create_dataloaders",
    "HybridQuantumCNN",
    "run_experiment",
]
