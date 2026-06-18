"""Classical CNN baseline — like-for-like match to HybridQuantumCNN.

The ONLY difference from HybridQuantumCNN is the quantum pathway:
  QCNN:     quantum_input (48→8→4)  →  QuantumCircuitLayer (4 qubits, 2 layers)
  Baseline: quantum_input_embed (48→8)  →  classical_circuit_block (8→16→4, Tanh)

Everything else — classical_branch, quantum_projection, classifier, init values —
is structurally identical so that metric differences isolate the quantum contribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassicalCNNBaseline(nn.Module):
    """Matched classical counterpart to HybridQuantumCNN.

    Architectural decisions that mirror the QCNN exactly:
      - Same classical CNN encoder (conv layers, pooling, channels).
      - Same 8-dim intermediate embedding from the image pool features.
      - Classical MLP (8 → 16 → 4, Tanh) replaces the variational quantum
        circuit; Tanh mirrors the [-1, 1] range of Pauli-Z expectation values.
      - Same quantum_projection (Linear 4 → hidden_dim//2).
      - Same fusion dimension (hidden_dim + hidden_dim//2).
      - Same classifier head (Linear → ReLU → Dropout → Linear).
    """

    def __init__(
        self,
        num_classes: int,
        quantum_embedding_dim: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # ------------------------------------------------------------------ #
        # Classical encoder — identical to HybridQuantumCNN.classical_branch  #
        # ------------------------------------------------------------------ #
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

        # ------------------------------------------------------------------ #
        # Quantum-pathway input embedding — first part of quantum_input in     #
        # QCNN: pool → flatten → Linear(48 → 8) → ReLU                       #
        # ------------------------------------------------------------------ #
        self.quantum_input_embed = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(3 * 4 * 4, quantum_embedding_dim),   # 48 → 8
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------ #
        # Classical circuit block — replaces Linear(8→4) + QuantumLayer(4→4)  #
        # Input/output dimensions are unchanged (8 in, 4 out).                #
        # Tanh bounds output to [-1, 1], matching Pauli-Z expectation range.  #
        # ------------------------------------------------------------------ #
        self.classical_circuit_block = nn.Sequential(
            nn.Linear(quantum_embedding_dim, 16),   # 8 → 16
            nn.ReLU(inplace=True),
            nn.Linear(16, 4),                       # 16 → 4
            nn.Tanh(),                              # ≈ Pauli-Z range [-1, 1]
        )

        # ------------------------------------------------------------------ #
        # Quantum projection — identical to HybridQuantumCNN                  #
        # (n_qubits=4 → hidden_dim//2=32)                                     #
        # ------------------------------------------------------------------ #
        self.quantum_projection = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------ #
        # Classifier head — identical to HybridQuantumCNN                     #
        # fused dim = hidden_dim(64) + hidden_dim//2(32) = 96                 #
        # ------------------------------------------------------------------ #
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        classical_features = self.classical_branch(images)

        # Classical replacement for the quantum pathway
        embed = self.quantum_input_embed(images)             # (B, 8)
        circuit_out = self.classical_circuit_block(embed)   # (B, 4)
        projected = self.quantum_projection(circuit_out)    # (B, 32)

        fused = torch.cat([classical_features, projected], dim=1)  # (B, 96)
        return self.classifier(fused)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_param_comparison(qcnn_model: nn.Module, baseline_model: nn.Module) -> None:
    """Print trainable parameter counts for both models side by side."""
    q_total = count_parameters(qcnn_model)
    c_total = count_parameters(baseline_model)

    # Break down QCNN
    q_branch  = count_parameters(qcnn_model.classical_branch)
    q_qinput  = count_parameters(qcnn_model.quantum_input)
    q_qlayer  = count_parameters(qcnn_model.quantum_layer)
    q_qproj   = count_parameters(qcnn_model.quantum_projection)
    q_cls     = count_parameters(qcnn_model.classifier)

    # Break down baseline
    c_branch  = count_parameters(baseline_model.classical_branch)
    c_embed   = count_parameters(baseline_model.quantum_input_embed)
    c_circ    = count_parameters(baseline_model.classical_circuit_block)
    c_proj    = count_parameters(baseline_model.quantum_projection)
    c_cls     = count_parameters(baseline_model.classifier)

    print("=" * 65)
    print(f"{'Component':<35} {'QCNN':>12} {'Classical':>12}")
    print("-" * 65)
    print(f"{'classical_branch':<35} {q_branch:>12,} {c_branch:>12,}")
    print(f"{'quantum_input / embed':<35} {q_qinput:>12,} {c_embed:>12,}")
    print(f"{'quantum_layer / classical_circuit':<35} {q_qlayer:>12,} {c_circ:>12,}")
    print(f"{'quantum_projection':<35} {q_qproj:>12,} {c_proj:>12,}")
    print(f"{'classifier':<35} {q_cls:>12,} {c_cls:>12,}")
    print("-" * 65)
    print(f"{'TOTAL trainable parameters':<35} {q_total:>12,} {c_total:>12,}")
    print("=" * 65)
