# Concrete Crack Detection - Final Year Project

This repository is dedicated to compiling, storing and developing all code related to the topic of concrete crack image classification. Herein, will be included different algorithms, approaches and findings from pertinent fields.

## 🎯 Project Overview

This project implements a comprehensive framework for detecting and classifying cracks in concrete structures using:
- **Deep Learning Models**: Neural networks for classification and segmentation
- **YOLO Algorithms**: State-of-the-art object detection (YOLOv8+)
- **Mask Generation**: Automated crack segmentation and mask creation
- **Digital Twin Integration**: Foundation for 3D visualization (planned)

## 🚀 Features

- ✅ **Dataset Management**: Easy import and organization of crack datasets
- ✅ **Pre-trained Models**: Load and use existing neural network models
- ✅ **YOLO Integration**: YOLOv8 for crack detection and classification
- ✅ **Mask Generation**: Multiple methods (threshold, edge detection, deep learning, hybrid)
- ✅ **Batch Processing**: Process multiple images efficiently
- ✅ **Visualization Tools**: Comprehensive visualization of results
- ✅ **Extensible Architecture**: Easy to add new models and methods

## 📁 Project Structure

```
Final_Year_Project/
├── src/
│   ├── data/              # Dataset loading and management
│   ├── models/            # Model loaders and architectures
│   ├── inference/         # Mask generation and inference
│   └── utils/             # Preprocessing and visualization utilities
├── config/
│   └── config.yaml        # Main configuration file
├── examples/
│   ├── basic_workflow.py  # Basic usage demonstration
│   ├── train_yolo.py      # YOLO training example
│   └── batch_inference.py # Batch processing example
├── datasets/              # Place your datasets here
│   ├── train/
│   ├── val/
│   └── test/
├── models/                # Saved models
└── outputs/               # Results and visualizations
```

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/antongl6462/Final_Year_Project.git
cd Final_Year_Project
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 📊 Dataset Preparation

### Directory Structure
Organize your crack images in the following structure:
```
datasets/
├── train/
│   ├── images/  # Training images
│   └── labels/  # YOLO format labels (optional)
├── val/
│   ├── images/  # Validation images
│   └── labels/
└── test/
    └── images/  # Test images
```

### YOLO Format Labels
For YOLO training, each image needs a corresponding `.txt` file with:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized to [0, 1].

## 🎓 Usage Examples

### 1. Basic Workflow
```python
from src.data.dataset_loader import CrackDataset
from src.models.yolo_detector import YOLOCrackDetector
from src.inference.mask_generator import MaskGenerator

# Initialize YOLO detector
yolo = YOLOCrackDetector(model_type='yolov8n.pt')

# Detect cracks
detections = yolo.detect_cracks('path/to/image.jpg', conf_threshold=0.25)

# Generate mask
mask_gen = MaskGenerator()
mask = mask_gen.generate_mask_threshold(image, method='otsu')
```

### 2. Run Example Scripts
```bash
# Basic workflow demo
python examples/basic_workflow.py

# Train YOLO model
python examples/train_yolo.py

# Batch inference
python examples/batch_inference.py
```

### 3. Training a Custom YOLO Model
```python
from src.models.yolo_detector import YOLOCrackDetector

yolo = YOLOCrackDetector(model_type='yolov8n.pt')
yolo.train(
    data_yaml='config/crack_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='crack_detection'
)
```

## 🔬 Model Architectures

### Available Models
1. **YOLO (Ultralytics YOLOv8)**
   - Detection: `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`
   - Segmentation: `yolov8n-seg.pt`, `yolov8s-seg.pt`, etc.

2. **Classification Backbones**
   - ResNet50, ResNet101
   - VGG16
   - EfficientNet-B0

3. **Segmentation**
   - U-Net (custom implementation)
   - Pre-trained models can be loaded

## 🎨 Mask Generation Methods

1. **Threshold-based**: Otsu, Adaptive, Binary thresholding
2. **Edge-based**: Canny edge detection
3. **Model-based**: Deep learning segmentation models
4. **Hybrid**: Combination of multiple methods

## 📈 Results and Visualization

The framework provides comprehensive visualization tools:
- Bounding box overlays for detections
- Mask overlays on original images
- Comparison grids
- Training history plots

Results are saved to `outputs/` directory.

## 🔮 Future Enhancements

- [ ] Digital twin integration for 3D visualization
- [ ] Real-time crack detection from video streams
- [ ] Mobile deployment (ONNX, TensorFlow Lite)
- [ ] Advanced crack severity assessment
- [ ] Automated report generation
- [ ] Web interface for easy access

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📝 License

This project is part of a Final Year Project. Please contact the repository owner for licensing information.

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact the repository owner.

## 🙏 Acknowledgments

- Ultralytics for YOLOv8 implementation
- PyTorch and torchvision teams
- OpenCV community
- Various concrete crack datasets used for research

---

**Note**: This project is actively under development. Features and documentation will be updated regularly. 
