"""Paper-inspired quantum circuit layer using PennyLane + PyTorch.

Architecture follows the paper's quantum convolutional layer:
  1. Angle encoding: Hadamard + RY(input_i) on each qubit.
  2. Trainable entanglement block: CNOT ring + RZ + RY per layer.
  3. Measurement: Pauli-Z expectation values returned as features.
"""

from __future__ import annotations

from typing import List

try:
    import pennylane as qml
except ImportError as exc:
    raise ImportError("PennyLane is required: pip install pennylane") from exc

import torch
import torch.nn as nn


class QuantumCircuitLayer(nn.Module):
    """Reusable quantum circuit layer compatible with PyTorch autograd.

    Args:
        n_qubits:  Number of qubits (equals input feature dimension).
        n_q_layers: Number of trainable entanglement blocks.
    """

    def __init__(self, n_qubits: int = 2, n_q_layers: int = 1) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_q_layers = n_q_layers

        # PennyLane device — default.qubit is a CPU simulator.
        self.device = qml.device("default.qubit", wires=n_qubits)

        # Trainable parameters: 2 rotations (RZ + RY) per qubit per layer.
        # Shape: [n_q_layers, n_qubits, 2]
        self.theta = nn.Parameter(
            torch.randn(n_q_layers, n_qubits, 2) * 0.01,
            requires_grad=True,
        )

        # Build quantum node and wrap as TorchLayer.
        weight_shapes = {"theta": (n_q_layers, n_qubits, 2)}
        qnode = qml.QNode(self._circuit, self.device, interface="torch", diff_method="backprop")
        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def _circuit(self, inputs: torch.Tensor, theta: torch.Tensor) -> List:
        """Quantum circuit definition.

        Args:
            inputs: 1-D tensor of length n_qubits (one sample).
            theta:  Trainable angles, shape [n_q_layers, n_qubits, 2].

        Returns:
            List of Pauli-Z expectation values, length n_qubits.
        """
        # --- Encoding stage ---
        # Hadamard + angle encoding per qubit.
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
            qml.RY(inputs[i], wires=i)

        # --- Trainable entanglement stage ---
        for layer_idx in range(self.n_q_layers):
            # CNOT ring entanglement.
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            # Per-qubit rotations.
            for i in range(self.n_qubits):
                qml.RZ(theta[layer_idx, i, 0], wires=i)
                qml.RY(theta[layer_idx, i, 1], wires=i)

        # --- Measurement stage ---
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [batch_size, n_qubits].

        Returns:
            Tensor of shape [batch_size, n_qubits] — Pauli-Z expectations.
        """
        if x.ndim != 2:
            raise ValueError(f"Expected 2-D input [batch, n_qubits], got {x.shape}")
        if x.shape[1] != self.n_qubits:
            raise ValueError(
                f"Input feature dim must equal n_qubits={self.n_qubits}, got {x.shape[1]}"
            )
        # PennyLane TorchLayer expects a 1-D input tensor per sample.
        outputs = [self.quantum_layer(sample) for sample in x]
        return torch.stack(outputs, dim=0)
