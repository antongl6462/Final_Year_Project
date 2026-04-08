# Hybrid QCNN Extension (Paper-Inspired)

## Purpose
This module adds a **separate hybrid quantum-classical experiment** inspired by:

> Recognition and Classification of Concrete Surface Cracks with an Inception Quantum Convolutional Neural Network Algorithm

It is implemented as a **proof-of-concept simulation** using PyTorch + PennyLane and does **not** replace the existing YOLO baseline.

## Design Summary
- Reuses existing split logic (`src/prepare_classify_data.py`) and existing prepared split directory (`classify_yolo/`) when available.
- Adds optional quantum-only preprocessing (grayscale, CLAHE, optional black-hat).
- Uses compact quantum embedding and low-qubit circuit for realistic runtime.
- Fuses classical and quantum features in a small hybrid classifier.

## Scientific Caution
- No claim of quantum advantage is made.
- Results should be interpreted as exploratory and methodological.
- Circuit size is intentionally compact to support local CPU simulation.

## Key Files
- `src/quantum_cnn/config.py`
- `src/quantum_cnn/data.py`
- `src/quantum_cnn/quantum_preprocessing.py`
- `src/quantum_cnn/quantum_layer.py`
- `src/quantum_cnn/hybrid_qcnn.py`
- `src/quantum_cnn/train_quantum_cnn.py`
- `src/quantum_cnn/run_experiment.py`
- `src/train_hybrid_qcnn.py`
- `src/compare_classical_vs_qcnn.py`
- `configs/hybrid_qcnn.json`

## Run
```bash
source quantum_cnn/.venv/bin/activate
python src/train_hybrid_qcnn.py --config configs/hybrid_qcnn.json --experiment_name hybrid_qcnn_fast
```

Artifacts are saved to `runs/quantum_cnn/<experiment_name>/`.
