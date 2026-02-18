#!/usr/bin/env python3
"""
Prepare YOLO classification dataset from raw classification images.
Expects input structure:
    DATA_ROOT/raw_classification/
        cracked/
        not_cracked/

Creates YOLO classification folder structure with stratified split:
    DATA_ROOT/classify_yolo/
        train/{cracked,not_cracked}/
        val/{cracked,not_cracked}/
        test/{cracked,not_cracked}/
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare YOLO classification dataset")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing raw_classification/ folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for YOLO format (default: DATA_ROOT/classify_yolo)",
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
        help="Use symlinks instead of copying files (faster, saves space)",
    )
    parser.add_argument(
        "--class_names",
        nargs=2,
        default=["cracked", "not_cracked"],
        help="Class names in order (default: cracked not_cracked)",
    )
    return parser.parse_args()


def get_image_paths(root_dir: Path, class_name: str) -> List[Path]:
    """Get all image paths for a given class."""
    class_dir = root_dir / class_name
    if not class_dir.exists():
        raise ValueError(f"Class directory not found: {class_dir}")
    
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = [p for p in class_dir.iterdir() if p.suffix.lower() in extensions]
    return sorted(images)


def stratified_split(
    files: List[Path],
    labels: List[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[Tuple[List[Path], List[int]], ...]:
    """Perform stratified split into train/val/test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # First split: train vs (val+test)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        files,
        labels,
        train_size=train_ratio,
        stratify=labels,
        random_state=seed,
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files,
        temp_labels,
        train_size=val_size,
        stratify=temp_labels,
        random_state=seed,
    )
    
    return (
        (train_files, train_labels),
        (val_files, val_labels),
        (test_files, test_labels),
    )


def create_yolo_structure(
    output_dir: Path,
    splits: Dict[str, Tuple[List[Path], List[int]]],
    class_names: List[str],
    use_symlink: bool = False,
) -> Dict:
    """Create YOLO classification folder structure and copy/link files."""
    
    # Create directory structure
    for split_name in splits.keys():
        for class_name in class_names:
            (output_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)
    
    # Track split metadata
    metadata = {
        "class_names": class_names,
        "splits": {},
    }
    
    # Copy or symlink files
    for split_name, (files, labels) in splits.items():
        split_metadata = {class_name: [] for class_name in class_names}
        
        pbar = tqdm(
            zip(files, labels),
            total=len(files),
            desc=f"Processing {split_name}",
        )
        
        for file_path, label in pbar:
            class_name = class_names[label]
            dest_path = output_dir / split_name / class_name / file_path.name
            
            if use_symlink:
                if dest_path.exists() or dest_path.is_symlink():
                    dest_path.unlink()
                dest_path.symlink_to(file_path.resolve())
            else:
                shutil.copy2(file_path, dest_path)
            
            split_metadata[class_name].append(str(file_path))
        
        metadata["splits"][split_name] = {
            "total": len(files),
            "per_class": {cn: len(split_metadata[cn]) for cn in class_names},
        }
    
    return metadata


def main():
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Resolve paths
    data_root = Path(args.data_root).resolve()
    raw_dir = data_root / "raw_classification"
    
    if not raw_dir.exists():
        raise ValueError(
            f"raw_classification directory not found at {raw_dir}\n"
            "Expected structure:\n"
            "  DATA_ROOT/raw_classification/\n"
            "    cracked/\n"
            "    not_cracked/"
        )
    
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "classify_yolo"
    output_dir = output_dir.resolve()
    
    print(f"Data root: {data_root}")
    print(f"Raw directory: {raw_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Class names: {args.class_names}")
    print(f"Split ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print(f"Random seed: {args.seed}")
    print(f"Mode: {'symlink' if args.symlink else 'copy'}")
    print()
    
    # Gather image paths
    all_files = []
    all_labels = []
    
    for label_idx, class_name in enumerate(args.class_names):
        print(f"Scanning class '{class_name}'...")
        images = get_image_paths(raw_dir, class_name)
        print(f"  Found {len(images)} images")
        all_files.extend(images)
        all_labels.extend([label_idx] * len(images))
    
    print(f"\nTotal images: {len(all_files)}")
    print(f"Class distribution: {np.bincount(all_labels)}")
    print()
    
    # Perform stratified split
    print("Performing stratified split...")
    (train_files, train_labels), (val_files, val_labels), (test_files, test_labels) = stratified_split(
        all_files,
        all_labels,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
    
    print(f"Train: {len(train_files)} images")
    print(f"Val: {len(val_files)} images")
    print(f"Test: {len(test_files)} images")
    print()
    
    # Create YOLO structure
    print("Creating YOLO classification structure...")
    splits = {
        "train": (train_files, train_labels),
        "val": (val_files, val_labels),
        "test": (test_files, test_labels),
    }
    
    metadata = create_yolo_structure(
        output_dir,
        splits,
        args.class_names,
        args.symlink,
    )
    
    # Save metadata
    metadata_file = data_root / "splits.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Dataset preparation complete!")
    print(f"  Output: {output_dir}")
    print(f"  Metadata: {metadata_file}")
    print(f"\nSplit summary:")
    for split_name, split_info in metadata["splits"].items():
        print(f"  {split_name}: {split_info['total']} images")
        for class_name, count in split_info["per_class"].items():
            print(f"    - {class_name}: {count}")


if __name__ == "__main__":
    main()
