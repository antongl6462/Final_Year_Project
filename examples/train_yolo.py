"""
Example script for training a YOLO model on custom crack dataset.
This demonstrates how to prepare data and train a YOLO model for crack detection.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.yolo_detector import YOLOCrackDetector
import yaml


def create_dataset_yaml(
    train_path: str,
    val_path: str,
    names: list,
    output_path: str = "config/crack_dataset.yaml"
):
    """
    Create YOLO dataset configuration file.
    
    Args:
        train_path: Path to training images
        val_path: Path to validation images
        names: List of class names
        output_path: Path to save YAML config
    """
    data = {
        'path': str(Path.cwd()),  # Dataset root
        'train': train_path,
        'val': val_path,
        'nc': len(names),  # Number of classes
        'names': names
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Dataset configuration saved to: {output_path}")
    return str(output_path)


def main():
    """Main training workflow."""
    
    print("=" * 60)
    print("YOLO Model Training for Crack Detection")
    print("=" * 60)
    
    # Step 1: Prepare dataset configuration
    print("\n1. Preparing dataset configuration...")
    
    # Define dataset structure
    # Expected format:
    # datasets/
    #   train/
    #     images/  (contains .jpg/.png files)
    #     labels/  (contains .txt files with YOLO format annotations)
    #   val/
    #     images/
    #     labels/
    
    dataset_yaml = create_dataset_yaml(
        train_path="datasets/train/images",
        val_path="datasets/val/images",
        names=['background', 'crack'],  # Modify based on your classes
        output_path="config/crack_dataset.yaml"
    )
    
    print("   ✓ Dataset configuration created")
    print("\n   Important: Ensure your dataset follows YOLO format:")
    print("   - Images in datasets/train/images/ and datasets/val/images/")
    print("   - Labels in datasets/train/labels/ and datasets/val/labels/")
    print("   - Each label file should have format: <class> <x_center> <y_center> <width> <height>")
    
    # Step 2: Initialize YOLO detector
    print("\n2. Initializing YOLO model...")
    yolo = YOLOCrackDetector(model_type='yolov8n.pt')  # Start with nano model
    print("   ✓ YOLO model initialized")
    
    # Step 3: Training configuration
    print("\n3. Training configuration:")
    epochs = 100
    img_size = 640
    batch_size = 16
    
    print(f"   - Epochs: {epochs}")
    print(f"   - Image size: {img_size}x{img_size}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Model: YOLOv8 Nano (upgrade to 's', 'm', or 'l' for better accuracy)")
    
    # Step 4: Start training
    print("\n4. Starting training...")
    print("   Note: Training can take several hours depending on dataset size")
    print("   Training logs will be saved to 'runs/detect/crack_detection'")
    
    try:
        # Check if dataset exists before training
        if not Path(dataset_yaml).exists():
            print("\n   ⚠ Dataset YAML not found!")
            print("   Please ensure dataset is properly set up before training")
            return
        
        # Uncomment to actually start training:
        # results = yolo.train(
        #     data_yaml=dataset_yaml,
        #     epochs=epochs,
        #     imgsz=img_size,
        #     batch=batch_size,
        #     name="crack_detection"
        # )
        
        print("\n   Training command prepared (uncomment in script to run)")
        print("   After training, the best model will be saved to:")
        print("   'runs/detect/crack_detection/weights/best.pt'")
        
    except Exception as e:
        print(f"\n   Error during training: {e}")
        print("   Make sure your dataset is properly formatted")
    
    # Step 5: Post-training steps
    print("\n5. After training:")
    print("   - Evaluate model performance on test set")
    print("   - Export model to ONNX for faster inference")
    print("   - Use the trained model for inference on new images")
    
    print("\n" + "=" * 60)
    print("Training workflow complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
