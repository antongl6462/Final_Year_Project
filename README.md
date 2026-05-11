# Concrete Crack Detection with YOLOv8 (Classification + Hybrid QCNN)

End-to-end project for concrete crack image classification using a strong classical baseline (**YOLOv8 classification**) and an optional **hybrid quantum-classical CNN (QCNN) extension** for research comparison.

## Table of Contents
- [Project Highlights](#project-highlights)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Data Setup](#data-setup)
- [Classical Pipeline (YOLOv8)](#classical-pipeline-yolov8)
- [Notebook Workflow](#notebook-workflow)
- [Hybrid QCNN Extension ](#hybrid-qcnn-extension-optional)
- [Outputs and Artifacts](#outputs-and-artifacts)
- [Troubleshooting](#troubleshooting)
- [Reproducibility Notes](#reproducibility-notes)

## Project Highlights
- Binary classification of concrete surface condition: `cracked` vs `not_cracked`
- Complete CLI pipeline for preparation, training, and evaluation
- Leakage-aware dataset workflow (hash-based duplicate handling and split auditing in notebook)
- Rich evaluation outputs: confusion matrix, ROC, PR, threshold sweep, JSON metrics
- Optional hybrid QCNN proof-of-concept experiment for methodological comparison

## Repository Structure
```text
Final_Year_Project/
├── README.md
├── DATA_SETUP_GUIDE.md
├── requirements.txt
├── setup_data.py
├── YOLO_Concrete_Crack_Detection.ipynb
├── concrete-crack-images-for-classification/   # Kaggle-style dataset (Positive/Negative)
├── raw_classification/                          # Canonical class folders (cracked/not_cracked)
├── classify_yolo/                               # YOLO split dataset (train/val/test)
├── runs/                                        # Training and evaluation outputs
├── configs/
│   └── hybrid_qcnn.json
├── docs/
│   └── qcnn_experiment.md
├── quantum_cnn/
│   ├── README.md
│   ├── requirements.txt
│   └── notebooks/
└── src/
    ├── prepare_classify_data.py
    ├── train_yolo_classify.py
    ├── eval_classify.py
    ├── train_hybrid_qcnn.py
    ├── compare_classical_vs_qcnn.py
    └── quantum_cnn/
```

## Quick Start

### 1) Create environment and install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Ensure data is available
If your dataset is already present in `raw_classification/`, skip to step 3.

To download/setup data automatically:
```bash
python setup_data.py --method kaggle --output_dir .
```

### 3) Prepare YOLO split folders
```bash
python src/prepare_classify_data.py --data_root . --symlink
```

### 4) Train YOLO classifier
```bash
python src/train_yolo_classify.py \
  --data_root . \
  --runs_dir ./runs \
  --model yolov8n-cls.pt \
  --epochs 20 \
  --patience 5 \
  --imgsz 224 \
  --batch 32 \
  --exist_ok
```

### 5) Evaluate trained model
```bash
python src/eval_classify.py \
  --model runs/classify/train/weights/best.pt \
  --data_root . \
  --split test \
  --threshold_sweep \
  --save_predictions
```

## Data Setup

Supported setup methods via `setup_data.py`:
- `--method kaggle`: downloads `arunrk7/surface-crack-detection`
- `--method zip`: extracts from a local zip
- `--method manual`: links an existing dataset directory

Expected canonical class structure after setup:
```text
raw_classification/
├── cracked/
└── not_cracked/
```

For full details, see `DATA_SETUP_GUIDE.md`.

## Classical Pipeline (YOLOv8)

### A) Prepare split dataset
`src/prepare_classify_data.py` creates:
```text
classify_yolo/
├── train/{cracked,not_cracked}/
├── val/{cracked,not_cracked}/
└── test/{cracked,not_cracked}/
```

Key options:
- `--train_ratio`, `--val_ratio`, `--test_ratio`
- `--seed` for reproducibility
- `--symlink` for fast, disk-efficient dataset creation

### B) Train model
`src/train_yolo_classify.py` supports core hyperparameters:
- `--model` (`yolov8n-cls.pt`, `yolov8s-cls.pt`, `yolov8m-cls.pt`, ...)
- `--epochs`, `--patience` (early stopping)
- `--imgsz`, `--batch`, `--optimizer`, `--lr0`, `--lrf`, `--weight_decay`
- `--device` (leave empty for auto-select)

### C) Evaluate model
`src/eval_classify.py` computes:
- Accuracy and per-class precision/recall/F1
- Confusion matrix (normalized plot)
- ROC-AUC + ROC curve (binary)
- Precision-Recall curve
- Optional threshold sweep analysis

## Notebook Workflow

Main notebook: `YOLO_Concrete_Crack_Detection.ipynb`

What it covers:
- Environment checks and package setup
- Data verification and split creation
- Leakage audit checks across train/val/test
- YOLO classification training with early stopping
- Visual diagnostics (loss, accuracy, confusion matrix, ROC, examples)
- Speed/size comparisons across model variants

> Note for Apple Silicon users: some `torchvision` operations (NMS) may fail on `mps` in certain setups. In this notebook, CPU inference/training is used where needed for compatibility.

## Hybrid QCNN Extension (Optional)

This repository includes a separate, paper-inspired hybrid QCNN proof-of-concept.

Run with:
```bash
python src/train_hybrid_qcnn.py --config configs/hybrid_qcnn.json
```

Compare classical YOLO vs QCNN (using an existing trained YOLO model):
```bash
python src/compare_classical_vs_qcnn.py \
  --data_root . \
  --classical_model runs/classify/train/weights/best.pt \
  --qcnn_config configs/hybrid_qcnn.json
```

Additional context:
- `docs/qcnn_experiment.md`
- `quantum_cnn/README.md`

## Outputs and Artifacts

Typical outputs:
- Training: `runs/classify/train/`
  - `weights/best.pt`, `weights/last.pt`
  - `results.csv`, `results.png`, confusion matrix images
- Evaluation: `runs/classify/train/eval/`
  - `metrics_test.json`
  - `confusion_matrix_test.png`
  - `roc_curve_test.png`
  - `pr_curve_test.png`
  - optional threshold sweep and predictions JSON

## Troubleshooting

### NumPy / PyTorch compatibility issues
If notebook reports NumPy interop errors, reinstall with pinned versions (as noted in notebook setup cells).

### Kaggle credentials not found
- Place `kaggle.json` at `~/.kaggle/kaggle.json`
- Run: `chmod 600 ~/.kaggle/kaggle.json`

### MPS `torchvision::nms` not implemented
Use CPU for the affected training/inference stages:
- CLI: pass `--device cpu`
- Notebook: set `device="cpu"` in inference/training cells where needed

### Low ROC-AUC / unstable metrics
- Increase training epochs and keep early stopping enabled
- Verify class balance and split integrity
- Run threshold sweep to inspect decision-threshold behavior
- Try a larger backbone (`yolov8s-cls.pt` / `yolov8m-cls.pt`)

## Reproducibility Notes
- Set fixed seeds (`--seed`) during split and training
- Keep split ratios and class order consistent
- Track exact package versions in your environment
- Save experiment configs and metrics per run
