# Quick Start Guide

Get started with concrete crack detection in 5 minutes!

## 🚀 Quick Setup

### 1. Install (2 minutes)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision opencv-python ultralytics numpy pillow matplotlib
```

### 2. Add Sample Images (1 minute)
```bash
# Create test directory
mkdir -p datasets/test

# Add your concrete crack images to datasets/test/
# Supported formats: .jpg, .jpeg, .png
```

### 3. Run Your First Detection (2 minutes)
```bash
# Run the basic workflow
python examples/basic_workflow.py

# Or try batch inference
python examples/batch_inference.py
```

## 📝 Basic Usage Example

```python
# Import modules
from src.models.yolo_detector import YOLOCrackDetector
from src.inference.mask_generator import MaskGenerator

# Initialize detector
yolo = YOLOCrackDetector(model_type='yolov8n.pt')

# Detect cracks in an image
detections = yolo.detect_cracks(
    'datasets/test/crack_image.jpg',
    conf_threshold=0.25
)

print(f"Found {len(detections)} cracks!")

# Generate segmentation mask
mask_gen = MaskGenerator()
mask = mask_gen.generate_mask_threshold(image, method='otsu')
```

## 🎯 Common Tasks

### Detect Cracks in Single Image
```python
from src.models.yolo_detector import YOLOCrackDetector

yolo = YOLOCrackDetector()
results = yolo.detect_cracks('path/to/image.jpg')
```

### Process Multiple Images
```python
from src.models.yolo_detector import YOLOCrackDetector

yolo = YOLOCrackDetector()
results = yolo.batch_inference('datasets/test', output_dir='outputs/results')
```

### Generate Crack Masks
```python
from src.inference.mask_generator import MaskGenerator
import cv2

mask_gen = MaskGenerator()
image = cv2.imread('crack_image.jpg')

# Using threshold method
mask = mask_gen.generate_mask_threshold(image, method='otsu')

# Using edge detection
mask = mask_gen.generate_mask_edges(image)

# Save mask
cv2.imwrite('crack_mask.png', mask)
```

### Train Custom YOLO Model
```python
from src.models.yolo_detector import YOLOCrackDetector

# Prepare your dataset in YOLO format first
# Then train:
yolo = YOLOCrackDetector(model_type='yolov8n.pt')
yolo.train(
    data_yaml='config/crack_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

## 📊 Understanding the Output

### Detection Output
```python
{
    'bbox': [x1, y1, x2, y2],      # Bounding box coordinates
    'confidence': 0.85,             # Detection confidence (0-1)
    'class_id': 1,                  # Class ID
    'class_name': 'crack'           # Class name
}
```

### Batch Results
Results are saved to `outputs/` directory:
```
outputs/
├── masks/              # Generated segmentation masks
├── detections/         # YOLO detection results
├── visualizations/     # Overlay images
└── results_summary.json
```

## ⚡ Performance Tips

### For Faster Inference
```python
# Use smaller YOLO model
yolo = YOLOCrackDetector(model_type='yolov8n.pt')  # Nano (fastest)

# For better accuracy, use larger models
yolo = YOLOCrackDetector(model_type='yolov8s.pt')  # Small
yolo = YOLOCrackDetector(model_type='yolov8m.pt')  # Medium
```

### For GPU Acceleration
```python
yolo = YOLOCrackDetector(model_type='yolov8n.pt', device='cuda')
```

### For CPU-Only Systems
```python
yolo = YOLOCrackDetector(model_type='yolov8n.pt', device='cpu')
```

## 🎓 Next Steps

1. ✅ **Explore Examples**: Check `examples/` directory for more advanced usage
2. ✅ **Customize Config**: Edit `config/config.yaml` for your needs
3. ✅ **Train Models**: Use your own dataset to train custom models
4. ✅ **Read Docs**: Check `README.md` for detailed documentation

## 🆘 Troubleshooting

### Import Errors
```bash
# Make sure you're in the project root directory
cd /path/to/Final_Year_Project

# Reinstall dependencies
pip install -r requirements.txt
```

### No Images Found
- Ensure images are in correct directory (datasets/test/)
- Check file extensions (.jpg, .png, .jpeg)
- Verify file permissions

### YOLO Model Download
- First run will download YOLO model (~6MB for yolov8n)
- Requires internet connection
- Models cached in `~/.ultralytics/`

## 📚 Learn More

- Full documentation: `README.md`
- Installation guide: `INSTALL.md`
- API documentation: See docstrings in source code

---

**Happy crack detecting! 🔍**
