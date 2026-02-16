"""
Example script for batch inference on a directory of images.
Demonstrates how to process multiple images and generate results.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import json
from src.models.yolo_detector import YOLOCrackDetector
from src.inference.mask_generator import MaskGenerator
from src.utils.visualization import visualize_detections, visualize_mask


def main():
    """Main batch inference workflow."""
    
    print("=" * 60)
    print("Batch Inference for Crack Detection")
    print("=" * 60)
    
    # Configuration
    input_dir = "datasets/test"  # Directory with images to process
    output_dir = "outputs/batch_inference"
    
    # Step 1: Setup
    print("\n1. Setting up inference pipeline...")
    
    # Initialize YOLO detector
    try:
        yolo = YOLOCrackDetector(model_type='yolov8n.pt')
        print("   ✓ YOLO detector initialized")
    except Exception as e:
        print(f"   Warning: Could not initialize YOLO: {e}")
        yolo = None
    
    # Initialize mask generator
    mask_gen = MaskGenerator()
    print("   ✓ Mask generator initialized")
    
    # Step 2: Find images
    print(f"\n2. Scanning for images in {input_dir}...")
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"   Error: Directory {input_dir} does not exist")
        print(f"   Creating directory...")
        input_path.mkdir(parents=True, exist_ok=True)
        print(f"   Please add images to {input_dir} and run again")
        return
    
    image_files = list(input_path.glob("*.jpg")) + \
                 list(input_path.glob("*.png")) + \
                 list(input_path.glob("*.jpeg"))
    
    print(f"   Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print(f"   No images found in {input_dir}")
        return
    
    # Step 3: Process images
    print("\n3. Processing images...")
    
    output_path = Path(output_dir)
    masks_dir = output_path / "masks"
    detections_dir = output_path / "detections"
    visualizations_dir = output_path / "visualizations"
    
    for dir_path in [masks_dir, detections_dir, visualizations_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n   Processing ({idx}/{len(image_files)}): {img_path.name}")
        
        # Load image
        image = cv2.imread(str(img_path))
        
        if image is None:
            print(f"   Error: Could not load {img_path}")
            continue
        
        result = {
            'image': img_path.name,
            'detections': [],
            'mask_generated': False
        }
        
        # Generate mask
        try:
            mask = mask_gen.generate_mask_threshold(image, method='otsu')
            mask_path = masks_dir / f"{img_path.stem}_mask.png"
            cv2.imwrite(str(mask_path), mask)
            result['mask_generated'] = True
            result['mask_path'] = str(mask_path)
            print(f"     ✓ Mask generated")
            
            # Create visualization
            overlay = visualize_mask(
                image, mask,
                save_path=str(visualizations_dir / f"{img_path.stem}_overlay.png"),
                show=False
            )
            
        except Exception as e:
            print(f"     Error generating mask: {e}")
        
        # YOLO detection
        if yolo:
            try:
                detections = yolo.detect_cracks(
                    str(img_path),
                    conf_threshold=0.25,
                    save_results=True,
                    output_dir=str(detections_dir)
                )
                result['detections'] = detections
                result['num_detections'] = len(detections)
                print(f"     ✓ YOLO detections: {len(detections)}")
                
            except Exception as e:
                print(f"     Error in YOLO detection: {e}")
        
        results.append(result)
    
    # Step 4: Save summary
    print("\n4. Saving results summary...")
    summary_path = output_path / "results_summary.json"
    
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   ✓ Summary saved to: {summary_path}")
    
    # Step 5: Statistics
    print("\n5. Processing statistics:")
    total_images = len(results)
    masks_generated = sum(1 for r in results if r['mask_generated'])
    total_detections = sum(r.get('num_detections', 0) for r in results)
    
    print(f"   - Total images processed: {total_images}")
    print(f"   - Masks generated: {masks_generated}")
    print(f"   - Total crack detections: {total_detections}")
    print(f"   - Average detections per image: {total_detections/max(total_images, 1):.2f}")
    
    print("\n" + "=" * 60)
    print("Batch inference complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
