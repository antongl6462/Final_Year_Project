"""Configuration schema for hybrid QCNN experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class QuantumCNNConfig:
    """All tunable settings for the hybrid QCNN experiment.

    The defaults are conservative so the experiment remains practical on a
    standard CPU-only machine.
    """

    dataset_root: Path = Path("./raw_classification")
    image_size: int = 48
    batch_size: int = 16
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42

    # Optional subsampling caps for runtime control.
    max_train_samples: Optional[int] = 1200
    max_val_samples: Optional[int] = 300
    max_test_samples: Optional[int] = 300

    # Quantum-only preprocessing controls.
    quantum_preprocess: bool = True
    use_grayscale: bool = True
    use_clahe: bool = True
    use_blackhat: bool = False

    # Model and optimization controls.
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 3
    n_qubits: int = 2
    n_q_layers: int = 1
    quantum_embedding_dim: int = 8
    hidden_dim: int = 64

    # Output controls.
    output_root: Path = Path("./runs/quantum_cnn")
    experiment_name: str = "hybrid_qcnn"

    @classmethod
    def from_json(cls, file_path: Path) -> "QuantumCNNConfig":
        """Build config from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        data["dataset_root"] = Path(data["dataset_root"])
        data["output_root"] = Path(data.get("output_root", "./runs/quantum_cnn"))
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a JSON-safe dictionary."""
        payload = asdict(self)
        payload["dataset_root"] = str(self.dataset_root)
        payload["output_root"] = str(self.output_root)
        return payload

    def save_json(self, file_path: Path) -> None:
        """Persist config to disk as JSON."""
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class QuantumCNNConfig:
    # Data and split settings
    dataset_root: Path = PROJECT_ROOT / "raw_classification"
    data_root_for_yolo_split: Path = PROJECT_ROOT
    use_existing_yolo_split: bool = True
    image_size: int = 64
    batch_size: int = 32
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    num_workers: int = 0
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None

    # Preprocessing settings (QCNN branch only)
    use_quantum_preprocessing: bool = True
    quantum_grayscale: bool = True
    quantum_crack_enhance: bool = True
    quantum_blur_ksize: int = 3
    quantum_clahe_clip: float = 2.0
    quantum_embedding_dim: int = 16

    # Training settings
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 8
    seed: int = 42
    optimizer: str = "Adam"

    # Quantum circuit settings
    n_qubits: int = 4
    n_q_layers: int = 2

    # Output and experiment tracking
    experiment_name: str = "hybrid_qcnn"
    output_root: Path = PROJECT_ROOT / "runs" / "quantum_cnn"

    @staticmethod
    def from_json(config_path: str | Path) -> "QuantumCNNConfig":
        """Load configuration values from a JSON file.

        Unknown keys are ignored so that old configs remain forward-compatible.
        """
        config_path = Path(config_path).resolve()
        with open(config_path, "r") as handle:
            raw: Dict[str, Any] = json.load(handle)

        valid_fields = {field.name for field in fields(QuantumCNNConfig)}
        filtered: Dict[str, Any] = {k: v for k, v in raw.items() if k in valid_fields}

        path_fields = {
            "dataset_root",
            "data_root_for_yolo_split",
            "output_root",
        }
        for key in path_fields:
            if key in filtered:
                filtered[key] = Path(filtered[key]).resolve()

        return QuantumCNNConfig(**filtered)

    def to_serializable_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable dictionary representation."""
        output: Dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            output[field.name] = str(value) if isinstance(value, Path) else value
        return output
