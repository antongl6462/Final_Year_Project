"""Configuration for hybrid QCNN experiments.

All hyperparameters are in one place so the experiment is reproducible
from a single JSON file, following the pattern of the classical pipeline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class QuantumCNNConfig:
    """Single source of truth for all hybrid QCNN settings.

    Conservative defaults allow a full run in ~1 minute on a standard CPU.
    Increase image_size / n_qubits / n_q_layers / epochs for heavier runs.
    """

    # ---- Data -------------------------------------------------------
    dataset_root: Path = Path("./raw_classification")
    """Root directory with cracked/ and not_cracked/ sub-folders."""

    image_size: int = 48
    """Input image side length (pixels). Small keeps simulation fast."""

    batch_size: int = 16
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42

    # ---- Runtime caps -----------------------------------------------
    max_train_samples: Optional[int] = 1200
    max_val_samples: Optional[int] = 300
    max_test_samples: Optional[int] = 300

    # ---- Quantum-only preprocessing (classical pipeline unaffected) --
    quantum_preprocess: bool = True
    use_grayscale: bool = True
    use_clahe: bool = True
    use_blackhat: bool = False

    # ---- Model and training -----------------------------------------
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 3

    n_qubits: int = 2
    """Number of qubits. Simulation cost grows exponentially with this."""

    n_q_layers: int = 1
    """Trainable entanglement blocks in the quantum circuit."""

    quantum_embedding_dim: int = 8
    """Intermediate dim before the quantum input layer."""

    hidden_dim: int = 64
    """Classical branch and fusion head width."""

    # ---- Output -----------------------------------------------------
    output_root: Path = Path("./runs/quantum_cnn")
    experiment_name: str = "hybrid_qcnn_fast"

    # ---- JSON helpers -----------------------------------------------
    @classmethod
    def from_json(cls, file_path: Path) -> "QuantumCNNConfig":
        """Load config from JSON; unknown keys are silently ignored."""
        with open(file_path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = json.load(fh)
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in valid}
        for key in ("dataset_root", "output_root"):
            if key in filtered:
                filtered[key] = Path(filtered[key])
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict."""
        payload = asdict(self)
        payload["dataset_root"] = str(self.dataset_root)
        payload["output_root"] = str(self.output_root)
        return payload

    def save_json(self, file_path: Path) -> None:
        """Persist config to disk."""
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
