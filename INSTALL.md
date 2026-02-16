# Installation and Setup Guide

## Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster training/inference

## Step-by-Step Installation

### 1. Clone the Repository
```bash
git clone https://github.com/antongl6462/Final_Year_Project.git
cd Final_Year_Project
```

### 2. Create Virtual Environment (Recommended)

**On Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

**Option A: Install all dependencies**
```bash
pip install -r requirements.txt
```

**Option B: Install minimal dependencies first**
```bash
# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy pillow matplotlib

# Then install YOLO and other tools
pip install ultralytics
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import torch; import cv2; from ultralytics import YOLO; print('✓ Installation successful')"
```

## Dataset Setup

### 1. Create Dataset Directories
```bash
mkdir -p datasets/train/images
mkdir -p datasets/train/labels
mkdir -p datasets/val/images
mkdir -p datasets/val/labels
mkdir -p datasets/test/images
```

### 2. Add Your Data
- Place training images in `datasets/train/images/`
- Place validation images in `datasets/val/images/`
- Place test images in `datasets/test/images/`

### 3. (Optional) Add YOLO Labels
If training YOLO models, add corresponding `.txt` label files in the `labels/` directories.

Label format (one line per object):
```
<class_id> <x_center> <y_center> <width> <height>
```
Where all values are normalized to [0, 1].

## Running Examples

### Test Basic Workflow
```bash
python examples/basic_workflow.py
```

### Run Batch Inference
```bash
# After adding images to datasets/test/
python examples/batch_inference.py
```

### Train YOLO Model
```bash
# After preparing labeled dataset
python examples/train_yolo.py
```

## Common Issues and Solutions

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution:** Install PyTorch
```bash
pip install torch torchvision
```

### Issue: "ModuleNotFoundError: No module named 'ultralytics'"
**Solution:** Install Ultralytics YOLO
```bash
pip install ultralytics
```

### Issue: CUDA out of memory
**Solution:** Reduce batch size in config or use CPU
```python
# In your code:
yolo = YOLOCrackDetector(model_type='yolov8n.pt', device='cpu')
```

### Issue: No images found in dataset
**Solution:** Ensure images are in correct directory and have proper extensions (.jpg, .png, .jpeg)

## GPU Setup (Optional)

### For NVIDIA GPUs
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Verify GPU
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Next Steps

1. ✅ Install all dependencies
2. ✅ Set up dataset directories
3. ✅ Add your crack images
4. ✅ Run basic workflow example
5. ✅ Train or load a model
6. ✅ Process your images
7. ✅ Analyze results

## Additional Resources

- **PyTorch Documentation**: https://pytorch.org/docs/
- **Ultralytics YOLO**: https://docs.ultralytics.com/
- **OpenCV Tutorials**: https://docs.opencv.org/

## Support

For issues or questions, please open an issue on GitHub.
