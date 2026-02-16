"""
Test script to verify project structure without requiring all dependencies.
"""

import os
import sys
from pathlib import Path


def check_directory_structure():
    """Check if all required directories exist."""
    required_dirs = [
        'src',
        'src/data',
        'src/models',
        'src/inference',
        'src/utils',
        'config',
        'examples'
    ]
    
    print("Checking directory structure...")
    all_exist = True
    for dir_path in required_dirs:
        exists = Path(dir_path).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_path}")
        if not exists:
            all_exist = False
    
    return all_exist


def check_required_files():
    """Check if all required files exist."""
    required_files = [
        'requirements.txt',
        'README.md',
        '.gitignore',
        'config/config.yaml',
        'src/__init__.py',
        'src/data/__init__.py',
        'src/data/dataset_loader.py',
        'src/models/__init__.py',
        'src/models/model_loader.py',
        'src/models/yolo_detector.py',
        'src/inference/__init__.py',
        'src/inference/mask_generator.py',
        'src/utils/__init__.py',
        'src/utils/preprocessing.py',
        'src/utils/visualization.py',
        'examples/basic_workflow.py',
        'examples/train_yolo.py',
        'examples/batch_inference.py'
    ]
    
    print("\nChecking required files...")
    all_exist = True
    for file_path in required_files:
        exists = Path(file_path).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file_path}")
        if not exists:
            all_exist = False
    
    return all_exist


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"\nPython version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("  ✓ Python version is compatible (3.8+)")
        return True
    else:
        print("  ✗ Python version should be 3.8 or higher")
        return False


def main():
    """Run all checks."""
    print("=" * 60)
    print("Project Structure Verification")
    print("=" * 60)
    
    checks = [
        check_python_version(),
        check_directory_structure(),
        check_required_files()
    ]
    
    print("\n" + "=" * 60)
    if all(checks):
        print("✓ All checks passed! Project structure is correct.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Add your dataset to datasets/ directory")
        print("3. Run examples: python examples/basic_workflow.py")
    else:
        print("✗ Some checks failed. Please review the output above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
