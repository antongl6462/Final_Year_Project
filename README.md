# Concrete Crack Detection - YOLO Pipeline

A complete, production-ready codebase for concrete crack detection using **Ultralytics YOLO**. Supports both **classification** (cracked vs not_cracked) and **segmentation** (crack masks) tasks.

> Important: segmentation is trained only on a **separate labeled segmentation dataset** (with true masks), not on the 40,000-image classification dataset.

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
├── segmentation_dataset.py         # Shared raster semantic dataset utilities
├── train_segmentation.py           # Train raster semantic segmentation model
├── evaluate_segmentation.py        # Evaluate semantic segmentation checkpoint
├── predict_segmentation.py         # Run semantic segmentation inference
├── utils_metrics.py                # Shared losses, metrics, threshold sweep
├── utils_visualisation.py          # Shared report figures and qualitative grids
├── setup_data.py                    # Automated dataset download/setup
├── requirements.txt                 # Python dependencies
├── DATA_SETUP_GUIDE.md             # Detailed data setup instructions
├── YOLO_Concrete_Crack_Detection.ipynb  # Interactive notebook
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

### Quick Start - Download Dataset

**Method 1: Automated Setup (Kaggle)**

```bash
# Install kaggle API
pip install kaggle

# Setup Kaggle credentials (one-time)
# 1. Go to https://www.kaggle.com/account
# 2. Create API token (downloads kaggle.json)
# 3. Place at ~/.kaggle/kaggle.json
# 4. Run: chmod 600 ~/.kaggle/kaggle.json

# Download and setup dataset
python setup_data.py --method kaggle --output_dir ../project-data
```

**Method 2: Manual Download**

```bash
# 1. Download dataset from Kaggle:
#    https://www.kaggle.com/datasets/arunrk7/surface-crack-detection
# 2. Extract the zip file
# 3. Run setup script:
python setup_data.py --method manual --data_path /path/to/extracted/data --output_dir ../project-data

# Or if you have the zip file:
python setup_data.py --method zip --zip_path /path/to/dataset.zip --output_dir ../project-data
```

**Method 3: Use Existing Data**

If you already have a concrete crack dataset:

```bash
# Dataset should have structure:
#   Positive/ or cracked/   (cracked images)
#   Negative/ or not_cracked/ (non-cracked images)

python setup_data.py --method manual --data_path /your/dataset/path --output_dir ../project-data
```

### Git Worktree (Optional)

The data branch (`copilot/import-datasets-and-models`) contains code utilities but not the dataset itself. If you want to access that branch:

```bash
# Check out data branch utilities
git worktree add ../project-data copilot/import-datasets-and-models
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

#### Segmentation Data (Optional, separate dataset)

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

This task uses a separate segmentation dataset with ground-truth masks. It is not run on the 40,000-image classification dataset.

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

### Step 4: Preferred Raster Semantic Segmentation Pipeline

For thin concrete cracks, the recommended workflow is now the raster-mask semantic pipeline rather than YOLO polygon supervision. It trains directly on binary masks, adds background-only negatives from the classification dataset, uses patch sampling for crack-heavy crops, and performs deterministic sliding-window inference on full-resolution images.

**New scripts:**
- `segmentation_dataset.py`: dataset discovery, quality checks, splits, patch extraction, sliding-window helpers
- `train_segmentation.py`: semantic training loop, checkpointing, validation threshold search, optional post-training evaluation
- `evaluate_segmentation.py`: threshold calibration, held-out test metrics, qualitative plots, worst-case analysis
- `predict_segmentation.py`: checkpoint inference on single images or folders
- `utils_metrics.py` / `utils_visualisation.py`: shared losses, metrics, post-processing, and report figures

#### Train the semantic model

```bash
python train_segmentation.py \
  --project_root . \
  --dataset_root raw_segmentation \
  --negative_dir concrete-crack-images-for-classification/Negative \
  --architecture unetplusplus \
  --encoder_name resnet34 \
  --patch_size 512 \
  --inference_overlap 0.5 \
  --epochs 40 \
  --batch_size 4 \
  --learning_rate 3e-4 \
  --positive_patch_prob 0.7 \
  --mixed_precision
```

**Training behaviour:**
- trains on raster masks directly instead of polygon labels
- prefers positive crack patches while still mixing in background-only crops
- uses `AdamW`, cosine scheduling, mixed precision on CUDA, and reproducible seeds
- saves checkpoints to `models/` and figures/tables to `outputs/`

#### Evaluate once on the held-out test split

```bash
python evaluate_segmentation.py \
  --checkpoint models/best_semantic_segmentation.pth \
  --dataset_root raw_segmentation \
  --negative_dir concrete-crack-images-for-classification/Negative
```

**Evaluation outputs:**
- `outputs/threshold_sweep.csv`: threshold search on validation predictions
- `outputs/segmentation_metrics_test.csv`: test-set metrics at the selected operating point
- `outputs/confusion_matrix.png`: pixel-level confusion matrix
- `outputs/qualitative_predictions.png`: representative overlays
- `outputs/worst_predictions.png`: failure-case panel for analysis

#### Run inference on new images

```bash
python predict_segmentation.py \
  --checkpoint models/best_semantic_segmentation.pth \
  --input raw_segmentation/images \
  --output_dir outputs/predictions
```

#### Why this pipeline is preferred

- crack masks remain in their original raster form, avoiding contour simplification loss
- threshold calibration is explicit, so recall-prioritised operating points can be reported honestly
- sliding-window inference preserves more native detail than aggressively shrinking full images
- post-processing is configurable and evaluated rather than assumed

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
- **Preferred Format**: Raster masks for semantic segmentation
- **Optional Baseline Format**: Converted YOLO polygon annotations

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
