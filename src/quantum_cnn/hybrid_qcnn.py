"""Hybrid quantum-classical classifier for crack recognition."""

from __future__ import annotations

import torch
import torch.nn as nn

from .quantum_layer import QuantumCircuitLayer


class HybridQuantumCNN(nn.Module):
    """Inception-style two-branch hybrid network.

    - Classical branch: shallow CNN encoder.
    - Quantum branch: compact embedding -> quantum circuit -> projection.
    - Fusion: concatenation + classifier head.
    """

    def __init__(
        self,
        num_classes: int,
        n_qubits: int,
        n_q_layers: int,
        quantum_embedding_dim: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.classical_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.quantum_input = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(3 * 4 * 4, quantum_embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(quantum_embedding_dim, n_qubits),
        )
        self.quantum_layer = QuantumCircuitLayer(n_qubits=n_qubits, n_q_layers=n_q_layers)
        self.quantum_projection = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        classical_features = self.classical_branch(images)
        quantum_inputs = self.quantum_input(images)
        quantum_features = self.quantum_layer(quantum_inputs)
        quantum_features = self.quantum_projection(quantum_features)

        fused = torch.cat([classical_features, quantum_features], dim=1)
        return self.classifier(fused)
