"""
Example script demonstrating basic crack detection workflow.
This script shows how to:
1. Load a dataset
2. Load a pre-trained model
3. Perform inference
4. Generate masks
5. Visualize results
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.data.dataset_loader import CrackDataset, DatasetImporter
from src.models.model_loader import ModelLoader, UNet
from src.models.yolo_detector import YOLOCrackDetector
from src.inference.mask_generator import MaskGenerator
from src.utils.preprocessing import ImagePreprocessor
from src.utils.visualization import visualize_detections, visualize_mask


def main():
    """Main demonstration workflow."""
    
    print("=" * 60)
    print("Concrete Crack Detection - Basic Workflow Demo")
    print("=" * 60)
    
    # Step 1: Check dataset availability
    print("\n1. Checking dataset structure...")
    dataset_path = "datasets/train"
    stats = DatasetImporter.verify_dataset_structure(dataset_path)
    print(f"   Dataset exists: {stats['exists']}")
    if stats['exists']:
        print(f"   Number of images: {stats['num_images']}")
        print(f"   Image formats: {stats['image_extensions']}")
    else:
        print(f"   Note: Dataset not found at {dataset_path}")
        print(f"   Please place your crack images in the 'datasets/train' directory")
    
    # Step 2: Initialize preprocessor
    print("\n2. Initializing image preprocessor...")
    preprocessor = ImagePreprocessor(target_size=(640, 640))
    print("   ✓ Preprocessor ready")
    
    # Step 3: Initialize mask generator
    print("\n3. Initializing mask generator...")
    # For demo, we'll use traditional CV methods (no model required)
    mask_gen = MaskGenerator()
    print("   ✓ Mask generator ready (using traditional CV methods)")
    
    # Optional: Load a UNet model if available
    # unet_model = UNet(in_channels=3, out_channels=1)
    # mask_gen = MaskGenerator(model=unet_model)
    
    # Step 4: Initialize YOLO detector
    print("\n4. Initializing YOLO detector...")
    print("   Note: This will download YOLOv8n model on first run")
    try:
        yolo = YOLOCrackDetector(model_type='yolov8n.pt')
        print("   ✓ YOLO detector ready")
    except Exception as e:
        print(f"   Warning: Could not initialize YOLO: {e}")
        yolo = None
    
    # Step 5: Demo with a sample image (if available)
    print("\n5. Processing demo...")
    
    # Check for sample images
    sample_dirs = [
        Path("datasets/train"),
        Path("datasets/test"),
        Path("examples")
    ]
    
    sample_image = None
    for sample_dir in sample_dirs:
        if sample_dir.exists():
            image_files = list(sample_dir.glob("*.jpg")) + \
                         list(sample_dir.glob("*.png"))
            if image_files:
                sample_image = str(image_files[0])
                break
    
    if sample_image:
        print(f"   Processing: {sample_image}")
        
        # Load image
        image = cv2.imread(sample_image)
        
        if image is not None:
            # Generate mask
            print("   - Generating mask...")
            mask = mask_gen.generate_mask_threshold(image, method='otsu')
            
            # Save results
            output_dir = Path("outputs/demo")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            mask_path = output_dir / "demo_mask.png"
            cv2.imwrite(str(mask_path), mask)
            print(f"   ✓ Mask saved to: {mask_path}")
            
            # Visualize (optional - requires display)
            # visualize_mask(image, mask, save_path=str(output_dir / "demo_overlay.png"))
            
            # YOLO detection (if available)
            if yolo:
                print("   - Running YOLO detection...")
                detections = yolo.detect_cracks(
                    sample_image,
                    conf_threshold=0.25,
                    save_results=True,
                    output_dir=str(output_dir)
                )
                print(f"   ✓ Found {len(detections)} detections")
        else:
            print(f"   Error: Could not load image from {sample_image}")
    else:
        print("   No sample images found in datasets/ or examples/")
        print("   Please add some concrete crack images to test the workflow")
    
    # Summary
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Add your crack dataset images to 'datasets/train' directory")
    print("2. Train a custom YOLO model on your dataset")
    print("3. Use the trained model for inference")
    print("4. Explore advanced features like digital twin integration")
    print("\nFor more examples, check the 'examples/' directory")
    print("=" * 60)


if __name__ == "__main__":
    main()
