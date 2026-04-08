"""PennyLane quantum layer used by the hybrid QCNN model."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import pennylane as qml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PennyLane is required for the QCNN extension. Install quantum_cnn/requirements.txt."
    ) from exc


class QuantumCircuitLayer(nn.Module):
    """Paper-inspired quantum feature extractor.

    Circuit stages:
    1) Hadamard + angle encoding via `RY`.
    2) Trainable entanglement blocks (`CNOT`, `RZ`, `RY`).
    3) Pauli-Z expectation measurements.

    Output shape: `[batch_size, n_qubits]`.
    """

    def __init__(self, n_qubits: int, n_q_layers: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_q_layers = n_q_layers

        device = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(device, interface="torch")
        def circuit(inputs, weights):
            for wire in range(n_qubits):
                qml.Hadamard(wires=wire)
                qml.RY(inputs[wire], wires=wire)

            for layer_index in range(n_q_layers):
                for wire in range(n_qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])
                if n_qubits > 1:
                    qml.CNOT(wires=[n_qubits - 1, 0])
                for wire in range(n_qubits):
                    qml.RZ(weights[layer_index, wire, 0], wires=wire)
                    qml.RY(weights[layer_index, wire, 1], wires=wire)

            return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

        weight_shapes = {"weights": (n_q_layers, n_qubits, 2)}
        self.layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with explicit shape guards.

        Args:
            x: Tensor shaped `[batch_size, n_qubits]`.
        """
        if x.ndim != 2:
            raise ValueError(f"Expected [batch, n_qubits], got {tuple(x.shape)}")
        if x.shape[1] != self.n_qubits:
            raise ValueError(
                f"Input feature size {x.shape[1]} does not match n_qubits={self.n_qubits}"
            )

        outputs = [self.layer(sample) for sample in x]
        return torch.stack(outputs, dim=0)
