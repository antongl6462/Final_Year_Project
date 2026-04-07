#!/usr/bin/env python3
"""
Prepare YOLO segmentation dataset from binary masks.

Expects input structure:
    DATA_ROOT/raw_segmentation/
        images/
        masks/   (binary masks: 0/1 or 0/255)

Creates YOLO segmentation format:
    DATA_ROOT/segment_yolo/
        images/{train,val,test}/
        labels/{train,val,test}/  (polygon annotations)
        crack-seg.yaml
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare YOLO segmentation dataset")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing raw_segmentation/ folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for YOLO format (default: DATA_ROOT/segment_yolo)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks for images instead of copying",
    )
    parser.add_argument(
        "--simplify_epsilon",
        type=float,
        default=0.001,
        help="Contour simplification epsilon (fraction of perimeter, default: 0.001)",
    )
    parser.add_argument(
        "--min_contour_area",
        type=int,
        default=50,
        help="Minimum contour area in pixels (default: 50)",
    )
    return parser.parse_args()


def get_paired_files(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    """Get paired image and mask files."""
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    if not masks_dir.exists():
        raise ValueError(f"Masks directory not found: {masks_dir}")
    
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    mask_extensions = {".png", ".jpg", ".bmp", ".tiff"}
    
    # Get all images
    images = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in image_extensions}
    
    # Find matching masks
    paired = []
    missing_masks = []
    
    for stem, image_path in sorted(images.items()):
        # Try to find mask with same stem
        mask_path = None
        for ext in mask_extensions:
            candidate = masks_dir / f"{stem}{ext}"
            if candidate.exists():
                mask_path = candidate
                break
        
        if mask_path:
            paired.append((image_path, mask_path))
        else:
            missing_masks.append(stem)
    
    if missing_masks:
        print(f"⚠ Warning: {len(missing_masks)} images have no matching mask")
        if len(missing_masks) <= 10:
            print(f"  Missing masks for: {missing_masks}")
    
    return paired


def mask_to_yolo_polygons(
    mask: np.ndarray,
    simplify_epsilon: float = 0.001,
    min_area: int = 50,
) -> List[np.ndarray]:
    """
    Convert binary mask to YOLO polygon format.
    
    Returns list of normalized polygons (N x 2), where coordinates are in [0, 1].
    """
    # Ensure binary mask (0 or 255)
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    height, width = mask.shape
    polygons = []
    
    for contour in contours:
        # Filter small contours
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Simplify contour
        epsilon = simplify_epsilon * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Need at least 3 points for a polygon
        if len(approx) < 3:
            continue
        
        # Reshape and normalize
        polygon = approx.reshape(-1, 2).astype(np.float32)
        polygon[:, 0] /= width  # Normalize x
        polygon[:, 1] /= height  # Normalize y
        
        # Clip to [0, 1]
        polygon = np.clip(polygon, 0, 1)
        
        polygons.append(polygon)
    
    return polygons


def create_yolo_label_file(
    polygons: List[np.ndarray],
    output_path: Path,
    class_id: int = 0,
):
    """Write YOLO format label file with polygon annotations."""
    with open(output_path, "w") as f:
        for polygon in polygons:
            # Format: class_id x1 y1 x2 y2 x3 y3 ...
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
            f.write(f"{class_id} {coords}\n")


def stratified_split_pairs(
    pairs: List[Tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, Path]], ...]:
    """Split paired files into train/val/test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Random split (stratification doesn't apply here unless we have class labels)
    indices = list(range(len(pairs)))
    
    # First split: train vs (val+test)
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_size,
        random_state=seed,
    )
    
    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    test_pairs = [pairs[i] for i in test_idx]
    
    return train_pairs, val_pairs, test_pairs


def process_split(
    pairs: List[Tuple[Path, Path]],
    split_name: str,
    output_dir: Path,
    use_symlink: bool,
    simplify_epsilon: float,
    min_area: int,
) -> Dict:
    """Process one data split (train/val/test)."""
    images_dir = output_dir / "images" / split_name
    labels_dir = output_dir / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "total": len(pairs),
        "with_annotations": 0,
        "total_polygons": 0,
        "skipped": 0,
    }
    
    pbar = tqdm(pairs, desc=f"Processing {split_name}")
    for image_path, mask_path in pbar:
        # Copy or symlink image
        dest_image = images_dir / image_path.name
        if use_symlink:
            if dest_image.exists() or dest_image.is_symlink():
                dest_image.unlink()
            dest_image.symlink_to(image_path.resolve())
        else:
            import shutil
            shutil.copy2(image_path, dest_image)
        
        # Read mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"⚠ Warning: Could not read mask {mask_path}")
            stats["skipped"] += 1
            continue
        
        # Convert to polygons
        polygons = mask_to_yolo_polygons(mask, simplify_epsilon, min_area)
        
        # Write label file
        label_path = labels_dir / f"{image_path.stem}.txt"
        create_yolo_label_file(polygons, label_path)
        
        if polygons:
            stats["with_annotations"] += 1
            stats["total_polygons"] += len(polygons)
    
    return stats


def create_yaml_file(output_dir: Path, class_names: List[str] = None):
    """Create YOLO dataset YAML file."""
    if class_names is None:
        class_names = ["crack"]
    
    yaml_content = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(class_names)},
        "nc": len(class_names),
    }
    
    yaml_path = output_dir / "crack-seg.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    return yaml_path


def main():
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Resolve paths
    data_root = Path(args.data_root).resolve()
    raw_dir = data_root / "raw_segmentation"
    
    if not raw_dir.exists():
        print(f"✗ Segmentation data not found at {raw_dir}")
        print("  Segmentation dataset preparation skipped.")
        print("  Expected structure:")
        print("    DATA_ROOT/raw_segmentation/")
        print("      images/")
        print("      masks/")
        return
    
    images_dir = raw_dir / "images"
    masks_dir = raw_dir / "masks"
    
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "segment_yolo"
    output_dir = output_dir.resolve()
    
    print("=" * 70)
    print("YOLO SEGMENTATION DATA PREPARATION")
    print("=" * 70)
    print(f"Data root: {data_root}")
    print(f"Raw directory: {raw_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print(f"Random seed: {args.seed}")
    print(f"Mode: {'symlink' if args.symlink else 'copy'}")
    print(f"Simplification epsilon: {args.simplify_epsilon}")
    print(f"Min contour area: {args.min_contour_area} pixels")
    print()
    
    # Get paired files
    print("Scanning for image-mask pairs...")
    pairs = get_paired_files(images_dir, masks_dir)
    print(f"✓ Found {len(pairs)} image-mask pairs")
    print()
    
    if len(pairs) == 0:
        print("✗ No valid image-mask pairs found. Exiting.")
        return
    
    # Split data
    print("Splitting data...")
    train_pairs, val_pairs, test_pairs = stratified_split_pairs(
        pairs,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
    
    print(f"  Train: {len(train_pairs)} pairs")
    print(f"  Val: {len(val_pairs)} pairs")
    print(f"  Test: {len(test_pairs)} pairs")
    print()
    
    # Process each split
    all_stats = {}
    for split_name, split_pairs in [
        ("train", train_pairs),
        ("val", val_pairs),
        ("test", test_pairs),
    ]:
        stats = process_split(
            split_pairs,
            split_name,
            output_dir,
            args.symlink,
            args.simplify_epsilon,
            args.min_contour_area,
        )
        all_stats[split_name] = stats
    
    print()
    
    # Create YAML file
    print("Creating dataset YAML...")
    yaml_path = create_yaml_file(output_dir)
    print(f"✓ YAML file created: {yaml_path}")
    print()
    
    # Save metadata
    metadata = {
        "class_names": ["crack"],
        "splits": all_stats,
        "config": vars(args),
    }
    
    metadata_file = data_root / "segment_splits.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("=" * 70)
    print("✓ Segmentation dataset preparation complete!")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Metadata: {metadata_file}")
    print(f"\nSummary:")
    for split_name, stats in all_stats.items():
        print(f"  {split_name}:")
        print(f"    Total images: {stats['total']}")
        print(f"    With annotations: {stats['with_annotations']}")
        print(f"    Total polygons: {stats['total_polygons']}")
        if stats['skipped'] > 0:
            print(f"    Skipped: {stats['skipped']}")
    
    print(f"\nYOLO config file: {yaml_path}")
    print("Use this path when training YOLO segmentation model.")


if __name__ == "__main__":
    main()
