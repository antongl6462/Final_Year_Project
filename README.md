
# Concrete Crack Detection: Classification & Segmentation

This repository provides a robust, modular PyTorch codebase for concrete surface image analysis, supporting both **image classification** (cracked vs not-cracked) and **semantic segmentation** (if masks are available). The codebase is designed for reproducibility, extensibility, and best practices in deep learning.

## Folder Structure

```
├── concrete-crack-images-for-classification/   # Dataset root (ImageFolder style)
│   ├── Negative/                              # Not-cracked images
│   └── Positive/                              # Cracked images
├── src/                                       # Core modules
│   ├── config.py
│   ├── data.py
│   ├── losses.py
│   ├── metrics.py
│   ├── models.py
│   ├── train_utils.py
│   └── viz.py
├── train_classification.py                    # Entrypoint: classification training
├── train_segmentation.py                      # Entrypoint: segmentation training
├── evaluate.py                                # Unified evaluation script
├── requirements.txt                           # All dependencies
├── README.md
```

## Setup

1. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	```
2. **Prepare dataset:**
	- Place images in `concrete-crack-images-for-classification/Negative` and `.../Positive`.
	- For segmentation, add `segmentation/images/` and `segmentation/masks/` (optional).

## Usage

**Train classification:**
```bash
python train_classification.py --data_root concrete-crack-images-for-classification --epochs 30 --batch_size 64
```

**Train segmentation:**
```bash
python train_segmentation.py --data_root concrete-crack-images-for-classification --epochs 30 --batch_size 16
```

**Evaluate model:**
```bash
python evaluate.py --data_root concrete-crack-images-for-classification --task classification --ckpt <path_to_ckpt>
python evaluate.py --data_root concrete-crack-images-for-classification --task segmentation --ckpt <path_to_ckpt>
```

## Features & Best Practices

- Stratified train/val/test splits for reproducibility
- Strong data augmentation (albumentations)
- AdamW optimizer, learning rate scheduling
- Early stopping, checkpointing, AMP, gradient clipping
- MixUp/CutMix (classification, optional)
- Modular code for easy extension
- Comprehensive metrics (F1, ROC-AUC, Dice, IoU)

## Anti-Overfitting Strategies

- **Augmentation:** Random crops, flips, color jitter, etc.
- **Regularization:** Dropout, weight decay, MixUp/CutMix
- **Early Stopping:** Monitors validation loss/metric
- **Learning Rate Scheduling:** Reduces LR on plateau
- **Stratified Splits:** Ensures balanced class distribution

## Reproducibility

- All scripts set random seeds and save splits/checkpoints for full reproducibility.

## Citation

If you use this codebase, please cite or acknowledge this repository.
