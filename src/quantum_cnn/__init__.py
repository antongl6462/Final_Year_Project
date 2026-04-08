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
from .config import QuantumCNNConfig
from .data import create_dataloaders
from .hybrid_qcnn import HybridQCNNClassifier
from .quantum_layer import PaperInspiredQuantumLayer
from .quantum_preprocessing import QuantumPreprocessor
from .run_experiment import run_experiment
from .train_quantum_cnn import train_model

try:
    from .quantum_model import HybridQuantumCNN
except ImportError:
    HybridQuantumCNN = None

__all__ = [
    "QuantumCNNConfig",
    "create_dataloaders",
    "QuantumPreprocessor",
    "PaperInspiredQuantumLayer",
    "HybridQCNNClassifier",
    "HybridQuantumCNN",
    "train_model",
    "run_experiment",
]
