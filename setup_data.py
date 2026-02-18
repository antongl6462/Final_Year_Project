#!/usr/bin/env python3
"""
Data Setup Script for Concrete Crack Detection

This script downloads and prepares the Concrete Crack Images dataset.
The dataset can be obtained from:
- Kaggle: https://www.kaggle.com/datasets/arunrk7/surface-crack-detection
- Alternative: Manual download and extraction

Usage:
    python setup_data.py --method kaggle
    python setup_data.py --method manual --data_path /path/to/downloaded/data
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path


def setup_from_kaggle(output_dir: Path):
    """Download dataset from Kaggle using kaggle API."""
    print("=" * 70)
    print("DOWNLOADING FROM KAGGLE")
    print("=" * 70)
    
    try:
        import kaggle
    except ImportError:
        print("✗ Kaggle API not installed")
        print("\nTo install:")
        print("  pip install kaggle")
        print("\nThen setup credentials:")
        print("  1. Go to https://www.kaggle.com/account")
        print("  2. Create API token (downloads kaggle.json)")
        print("  3. Place at ~/.kaggle/kaggle.json")
        print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    print("Downloading dataset from Kaggle...")
    print("Dataset: arunrk7/surface-crack-detection")
    
    try:
        # Download dataset
        kaggle.api.dataset_download_files(
            'arunrk7/surface-crack-detection',
            path=str(output_dir),
            unzip=True
        )
        print("✓ Download complete!")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("\nAlternatives:")
        print("  1. Download manually from Kaggle")
        print("  2. Use --method manual flag")
        return False


def setup_from_manual(data_path: Path, output_dir: Path):
    """Setup dataset from manually downloaded files."""
    print("=" * 70)
    print("SETTING UP FROM MANUAL DOWNLOAD")
    print("=" * 70)
    
    if not data_path.exists():
        print(f"✗ Data path not found: {data_path}")
        return False
    
    # Check for common dataset structures
    possible_structures = [
        (data_path / "Positive", data_path / "Negative"),
        (data_path / "cracked", data_path / "not_cracked"),
        (data_path / "Crack", data_path / "No_Crack"),
    ]
    
    source_positive = None
    source_negative = None
    
    for pos_dir, neg_dir in possible_structures:
        if pos_dir.exists() and neg_dir.exists():
            source_positive = pos_dir
            source_negative = neg_dir
            break
    
    if not source_positive:
        print("✗ Could not find valid dataset structure")
        print("\nExpected one of:")
        print("  - Positive/ and Negative/")
        print("  - cracked/ and not_cracked/")
        print("  - Crack/ and No_Crack/")
        return False
    
    print(f"✓ Found dataset structure:")
    print(f"  Positive: {source_positive}")
    print(f"  Negative: {source_negative}")
    
    # Create output structure
    raw_class_dir = output_dir / "raw_classification"
    raw_class_dir.mkdir(parents=True, exist_ok=True)
    
    dest_positive = raw_class_dir / "cracked"
    dest_negative = raw_class_dir / "not_cracked"
    
    print("\nCreating symlinks...")
    
    # Remove existing links if they exist
    if dest_positive.exists() or dest_positive.is_symlink():
        dest_positive.unlink()
    if dest_negative.exists() or dest_negative.is_symlink():
        dest_negative.unlink()
    
    # Create symlinks
    dest_positive.symlink_to(source_positive.resolve())
    dest_negative.symlink_to(source_negative.resolve())
    
    print("✓ Symlinks created!")
    return True


def setup_from_extract(zip_path: Path, output_dir: Path):
    """Extract and setup dataset from zip file."""
    print("=" * 70)
    print("EXTRACTING DATASET")
    print("=" * 70)
    
    if not zip_path.exists():
        print(f"✗ Zip file not found: {zip_path}")
        return False
    
    print(f"Extracting {zip_path}...")
    
    extract_dir = output_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("✓ Extraction complete!")
        
        # Now setup from the extracted directory
        return setup_from_manual(extract_dir, output_dir)
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False


def verify_setup(output_dir: Path):
    """Verify the dataset setup."""
    print("\n" + "=" * 70)
    print("VERIFYING SETUP")
    print("=" * 70)
    
    raw_class_dir = output_dir / "raw_classification"
    
    if not raw_class_dir.exists():
        print("✗ raw_classification directory not found")
        return False
    
    cracked_dir = raw_class_dir / "cracked"
    not_cracked_dir = raw_class_dir / "not_cracked"
    
    if not cracked_dir.exists():
        print("✗ cracked directory not found")
        return False
    
    if not not_cracked_dir.exists():
        print("✗ not_cracked directory not found")
        return False
    
    # Count images
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    cracked_images = [p for p in cracked_dir.iterdir() if p.suffix.lower() in extensions]
    not_cracked_images = [p for p in not_cracked_dir.iterdir() if p.suffix.lower() in extensions]
    
    print("✓ Dataset structure verified!")
    print(f"\n  Cracked images: {len(cracked_images)}")
    print(f"  Not cracked images: {len(not_cracked_images)}")
    print(f"  Total: {len(cracked_images) + len(not_cracked_images)}")
    
    if len(cracked_images) == 0 or len(not_cracked_images) == 0:
        print("\n⚠ Warning: One or both classes have no images")
        return False
    
    print(f"\n✓ Dataset ready at: {raw_class_dir}")
    return True


def print_manual_instructions():
    """Print manual download instructions."""
    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print("\n1. Download the Concrete Crack Images dataset:")
    print("   Option A: Kaggle")
    print("     URL: https://www.kaggle.com/datasets/arunrk7/surface-crack-detection")
    print("     Download: Click 'Download' button")
    print()
    print("   Option B: Alternative sources")
    print("     - Mendeley Data: Search for 'Concrete Crack Images for Classification'")
    print("     - Other repositories with crack detection datasets")
    print()
    print("2. Extract the downloaded zip file")
    print()
    print("3. Run this script with the extracted path:")
    print("   python setup_data.py --method manual --data_path /path/to/extracted/data")
    print()
    print("   Or if you have the zip file:")
    print("   python setup_data.py --method zip --zip_path /path/to/dataset.zip")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Setup Concrete Crack Detection Dataset")
    parser.add_argument(
        "--method",
        type=str,
        choices=["kaggle", "manual", "zip"],
        default="kaggle",
        help="Method to obtain dataset (default: kaggle)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to manually downloaded dataset (for manual method)",
    )
    parser.add_argument(
        "--zip_path",
        type=str,
        default=None,
        help="Path to dataset zip file (for zip method)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../project-data",
        help="Output directory (default: ../project-data)",
    )
    
    args = parser.parse_args()
    
    # Resolve output directory
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("CONCRETE CRACK DATASET SETUP")
    print("=" * 70)
    print(f"Method: {args.method}")
    print(f"Output directory: {output_dir}")
    print()
    
    success = False
    
    if args.method == "kaggle":
        success = setup_from_kaggle(output_dir)
        if not success:
            print_manual_instructions()
    
    elif args.method == "manual":
        if not args.data_path:
            print("✗ --data_path required for manual method")
            print_manual_instructions()
            return
        
        data_path = Path(args.data_path).resolve()
        success = setup_from_manual(data_path, output_dir)
    
    elif args.method == "zip":
        if not args.zip_path:
            print("✗ --zip_path required for zip method")
            print_manual_instructions()
            return
        
        zip_path = Path(args.zip_path).resolve()
        success = setup_from_extract(zip_path, output_dir)
    
    if success:
        verify_setup(output_dir)
        print("\n" + "=" * 70)
        print("✓ SETUP COMPLETE!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Run the Jupyter notebook: YOLO_Concrete_Crack_Detection.ipynb")
        print("  2. Or use the command-line scripts in src/")
        print()
    else:
        print("\n" + "=" * 70)
        print("✗ SETUP FAILED")
        print("=" * 70)
        print("\nPlease follow the manual instructions above.")


if __name__ == "__main__":
    main()
