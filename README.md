# Concrete Crack Detection - YOLO Pipeline

A complete, production-ready codebase for concrete crack detection using **Ultralytics YOLO**. Supports both **classification** (cracked vs not_cracked) and **segmentation** (crack masks) tasks.

## 🚀 Features

- **YOLO Classification**: Binary classification with F1/precision/recall/ROC-AUC evaluation
- **YOLO Segmentation**: Instance segmentation with pixel-level Dice/IoU metrics
- **Stratified splitting**: Reproducible train/val/test splits
- **Early stopping & regularization**: Prevent overfitting with patience and weight decay
- **Comprehensive evaluation**: Confusion matrices, ROC curves, threshold sweeps, visual overlays
- **Git worktree support**: Access data from different branches without mixing code and data

---

## 📁 Project Structure

```
Final_Year_Project/
├── src/
│   ├── prepare_classify_data.py    # Build YOLO classification dataset
│   ├── train_yolo_classify.py      # Train classification model
│   ├── eval_classify.py            # Evaluate classification model
│   ├── prepare_segment_data.py     # Build YOLO segmentation dataset
│   ├── train_yolo_segment.py       # Train segmentation model
│   └── eval_segment.py             # Evaluate segmentation model
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── runs/                           # Training outputs (auto-created)
```

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/antongl6462/Final_Year_Project.git
cd Final_Year_Project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**
- `ultralytics>=8.0.0` (YOLO)
- `opencv-python>=4.8.0`
- `numpy>=1.24.0`
- `scikit-learn>=1.3.0`
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`
- `torch>=2.0.0`
- `torchvision>=0.15.0`

---

## 🗂️ Data Setup

### Data lives in a separate git branch

This project supports accessing data from a **different git branch** to keep code and large datasets separate.

#### Option 1: Git Worktree (Recommended)

Use `git worktree` to check out the data branch into a separate directory:

```bash
# Check out data branch to ../project-data
git worktree add ../project-data copilot/import-datasets-and-models

# Verify data structure
ls ../project-data/
```

Then use `--data_root ../project-data` when running scripts.

#### Option 2: Manual Checkout

Alternatively, manually check out the data branch:

```bash
# Clone data branch separately
git clone -b copilot/import-datasets-and-models https://github.com/antongl6462/Final_Year_Project.git project-data
```

### Expected Data Structure

#### Classification Data

```
DATA_ROOT/raw_classification/
  cracked/          # Positive class images (227×227 RGB)
  not_cracked/      # Negative class images (227×227 RGB)
```

**Example:** If using worktree with `../project-data`:
```
../project-data/raw_classification/
  cracked/
  not_cracked/
```

Or if data is in `concrete-crack-images-for-classification/`:
```
concrete-crack-images-for-classification/
  Positive/   # Will be treated as 'cracked'
  Negative/   # Will be treated as 'not_cracked'
```

#### Segmentation Data (Optional)

```
DATA_ROOT/raw_segmentation/
  images/           # RGB images
  masks/            # Binary masks (0/255 or 0/1, same stem as images)
```

---

## 🎯 Task A: Classification

### Step 1: Prepare Data

Convert raw classification images to YOLO format with stratified split:

```bash
python src/prepare_classify_data.py \
  --data_root ../project-data \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 42
```

**Output:**
```
DATA_ROOT/classify_yolo/
  train/
    cracked/
    not_cracked/
  val/
    cracked/
    not_cracked/
  test/
    cracked/
    not_cracked/
DATA_ROOT/splits.json  # Split metadata
```

**Options:**
- `--symlink`: Use symlinks instead of copying (faster, saves space)
- `--class_names cracked not_cracked`: Customize class names

### Step 2: Train Model

Train YOLO classification model:

```bash
python src/train_yolo_classify.py \
  --data_root ../project-data \
  --runs_dir ./runs \
  --model yolov8n-cls.pt \
  --epochs 100 \
  --batch 32 \
  --imgsz 224 \
  --patience 20 \
  --weight_decay 0.0005 \
  --device 0 \
  --seed 42
```

**Key Parameters:**
- `--model`: YOLO model (`yolov8n-cls.pt`, `yolov8s-cls.pt`, `yolov8m-cls.pt`, etc.)
- `--epochs`: Maximum epochs (default: 100, uses early stopping)
- `--patience`: Early stopping patience (default: 20)
- `--weight_decay`: L2 regularization (default: 0.0005)
- `--imgsz`: Image size (default: 224, can use 256/320)
- `--device`: GPU device (0, 1, ...) or '' for auto, 'cpu' for CPU

**Output:** Model saved to `runs/classify/train/weights/best.pt`

### Step 3: Evaluate Model

Compute classification metrics:

```bash
python src/eval_classify.py \
  --model runs/classify/train/weights/best.pt \
  --data_root ../project-data \
  --split test \
  --device 0
```

**Output:**
- `runs/classify/train/eval/metrics_test.json`: Precision, recall, F1, ROC-AUC
- `runs/classify/train/eval/confusion_matrix_test.png`: Confusion matrix
- `runs/classify/train/eval/roc_curve_test.png`: ROC curve
- `runs/classify/train/eval/pr_curve_test.png`: Precision-recall curve

**Optional:**
- `--threshold_sweep`: Find optimal threshold by F1 score
- `--save_predictions`: Save all predictions to JSON

---

## 🎨 Task B: Segmentation

### Step 1: Prepare Data

Convert binary masks to YOLO polygon annotations:

```bash
python src/prepare_segment_data.py \
  --data_root ../project-data \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 42
```

**Output:**
```
DATA_ROOT/segment_yolo/
  images/
    train/
    val/
    test/
  labels/
    train/  # .txt files with polygon annotations
    val/
    test/
  crack-seg.yaml  # YOLO dataset config
DATA_ROOT/segment_splits.json  # Split metadata
```

**Options:**
- `--symlink`: Use symlinks for images
- `--simplify_epsilon 0.001`: Contour simplification (fraction of perimeter)
- `--min_contour_area 50`: Minimum contour area in pixels

### Step 2: Train Model

Train YOLO segmentation model:

```bash
python src/train_yolo_segment.py \
  --data_yaml ../project-data/segment_yolo/crack-seg.yaml \
  --runs_dir ./runs \
  --model yolov8n-seg.pt \
  --epochs 150 \
  --batch 16 \
  --imgsz 256 \
  --patience 30 \
  --weight_decay 0.0005 \
  --device 0 \
  --seed 42
```

**Key Parameters:**
- `--model`: YOLO segmentation model (`yolov8n-seg.pt`, `yolov8s-seg.pt`, etc.)
- `--epochs`: Maximum epochs (default: 150)
- `--patience`: Early stopping patience (default: 30)
- `--close_mosaic 10`: Disable mosaic augmentation in last N epochs
- `--mask_ratio 4`: Mask downsample ratio

**Output:** Model saved to `runs/segment/train/weights/best.pt`

### Step 3: Evaluate Model

Compute pixel-level Dice and IoU:

```bash
python src/eval_segment.py \
  --model runs/segment/train/weights/best.pt \
  --data_yaml ../project-data/segment_yolo/crack-seg.yaml \
  --split test \
  --device 0 \
  --save_worst 10 \
  --save_best 5
```

**Output:**
- `runs/segment/train/eval/metrics_test.json`: Dice, IoU statistics
- `runs/segment/train/eval/metrics_distribution_test.png`: Histogram of Dice/IoU
- `runs/segment/train/eval/worst_cases/`: Visual overlays of worst predictions
- `runs/segment/train/eval/best_cases/`: Visual overlays of best predictions

**Visualization Legend:**
- 🟢 Green: Ground truth only (false negative)
- 🔴 Red: Prediction only (false positive)
- 🟡 Yellow: Overlap (true positive)

---

## 🔧 Advanced Configuration

### Custom Training Parameters

#### Classification Example: High Regularization
```bash
python src/train_yolo_classify.py \
  --data_root ../project-data \
  --model yolov8s-cls.pt \
  --epochs 200 \
  --patience 30 \
  --weight_decay 0.001 \
  --dropout 0.2 \
  --optimizer AdamW \
  --lr0 0.001 \
  --lrf 0.001 \
  --cos_lr
```

#### Segmentation Example: Larger Model
```bash
python src/train_yolo_segment.py \
  --data_yaml ../project-data/segment_yolo/crack-seg.yaml \
  --model yolov8m-seg.pt \
  --epochs 300 \
  --batch 8 \
  --imgsz 320 \
  --patience 50 \
  --close_mosaic 20
```

### Evaluation Options

#### Classification with Threshold Sweep
```bash
python src/eval_classify.py \
  --model runs/classify/train/weights/best.pt \
  --data_root ../project-data \
  --split val \
  --threshold_sweep \
  --save_predictions
```

#### Segmentation with Custom Confidence
```bash
python src/eval_segment.py \
  --model runs/segment/train/weights/best.pt \
  --data_yaml ../project-data/segment_yolo/crack-seg.yaml \
  --conf 0.5 \
  --iou 0.5 \
  --save_worst 20
```

---

## 📊 Metrics Explained

### Classification Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correctness: (TP + TN) / Total |
| **Precision** | True positive rate: TP / (TP + FP) |
| **Recall** | Sensitivity: TP / (TP + FN) |
| **F1-Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under ROC curve (0.5=random, 1.0=perfect) |

### Segmentation Metrics

| Metric | Description |
|--------|-------------|
| **Dice** | 2 × Intersection / (Pred + GT), range [0, 1] |
| **IoU** | Intersection / Union, range [0, 1] |

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Data not found
```
ValueError: raw_classification directory not found
```
**Solution:** Check `--data_root` path. If using git worktree, verify:
```bash
ls ../project-data/raw_classification/
```

#### 2. GPU memory error
```
CUDA out of memory
```
**Solution:** Reduce batch size:
```bash
--batch 8  # or 4, 2
```

#### 3. No masks in segmentation data
```
✗ Segmentation data not found
```
**Solution:** Segmentation is optional. If you only have classification data, skip segmentation steps.

#### 4. Import errors
```
ModuleNotFoundError: No module named 'ultralytics'
```
**Solution:** Reinstall dependencies:
```bash
pip install -r requirements.txt
```

---

## 📚 Model Zoo

### Recommended Models

| Task | Model | Size | Speed | Accuracy |
|------|-------|------|-------|----------|
| Classification | `yolov8n-cls.pt` | Nano | ⚡️⚡️⚡️ | ⭐️⭐️⭐️ |
| Classification | `yolov8s-cls.pt` | Small | ⚡️⚡️ | ⭐️⭐️⭐️⭐️ |
| Classification | `yolov8m-cls.pt` | Medium | ⚡️ | ⭐️⭐️⭐️⭐️⭐️ |
| Segmentation | `yolov8n-seg.pt` | Nano | ⚡️⚡️⚡️ | ⭐️⭐️⭐️ |
| Segmentation | `yolov8s-seg.pt` | Small | ⚡️⚡️ | ⭐️⭐️⭐️⭐️ |
| Segmentation | `yolov8m-seg.pt` | Medium | ⚡️ | ⭐️⭐️⭐️⭐️⭐️ |

---

## 🔬 Dataset Information

### Classification Dataset
- **Total Images**: 40,000 (227×227 RGB)
- **Classes**: Cracked (50%) / Not Cracked (50%)
- **Default Split**: 80% train / 10% val / 10% test
- **Balanced**: Yes

### Segmentation Dataset (if available)
- **Images**: RGB images
- **Masks**: Binary (0=background, 255=crack)
- **Format**: Converted to YOLO polygon annotations

---

## 📖 Citation

If you use this codebase, please cite:

```bibtex
@misc{concrete_crack_yolo,
  author = {Anton},
  title = {Concrete Crack Detection with YOLO},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/antongl6462/Final_Year_Project}
}
```

---

## 📝 License

This project is open source and available under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the repository owner.

---

## 🙏 Acknowledgments

- **Ultralytics YOLO**: https://github.com/ultralytics/ultralytics
- **Dataset**: Concrete crack images for classification

---

## 🚦 Quick Start Summary

```bash
# 1. Setup
git worktree add ../project-data copilot/import-datasets-and-models
pip install -r requirements.txt

# 2. Classification
python src/prepare_classify_data.py --data_root ../project-data
python src/train_yolo_classify.py --data_root ../project-data --epochs 100
python src/eval_classify.py --model runs/classify/train/weights/best.pt --data_root ../project-data

# 3. Segmentation (optional)
python src/prepare_segment_data.py --data_root ../project-data
python src/train_yolo_segment.py --data_yaml ../project-data/segment_yolo/crack-seg.yaml --epochs 150
python src/eval_segment.py --model runs/segment/train/weights/best.pt --data_yaml ../project-data/segment_yolo/crack-seg.yaml
```

---

**Happy Training! 🎉** 
