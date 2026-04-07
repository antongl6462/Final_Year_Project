#!/usr/bin/env python3
"""
Train YOLO classification model for concrete crack detection.

Usage:
    python src/train_yolo_classify.py --data_root /path/to/data --runs_dir ./runs
"""

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO classification model")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing classify_yolo/ folder",
    )
    parser.add_argument(
        "--runs_dir",
        type=str,
        default="./runs",
        help="Directory to save training runs (default: ./runs)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-cls.pt",
        help="YOLO classification model (default: yolov8n-cls.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Image size for training (default: 224)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0005,
        help="Weight decay for regularization (default: 0.0005)",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final learning rate factor (default: 0.01)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use (default: '' for auto-select)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Project name (default: RUNS_DIR/classify)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Experiment name (default: train)",
    )
    parser.add_argument(
        "--exist_ok",
        action="store_true",
        help="Allow overwriting existing experiment",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained weights (default: True)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        choices=["SGD", "Adam", "AdamW"],
        help="Optimizer (default: SGD)",
    )
    parser.add_argument(
        "--cos_lr",
        action="store_true",
        default=True,
        help="Use cosine learning rate scheduler (default: True)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability (default: 0.0)",
    )
    return parser.parse_args()


def validate_data_structure(data_root: Path):
    """Validate YOLO classification data structure."""
    classify_dir = data_root / "classify_yolo"
    
    if not classify_dir.exists():
        raise ValueError(
            f"YOLO classification directory not found: {classify_dir}\n"
            "Please run prepare_classify_data.py first."
        )
    
    required_dirs = ["train", "val"]
    for split in required_dirs:
        split_dir = classify_dir / split
        if not split_dir.exists():
            raise ValueError(f"Required split directory not found: {split_dir}")
        
        # Check for at least one class folder
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        if not class_dirs:
            raise ValueError(f"No class directories found in {split_dir}")
    
    return classify_dir


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Resolve paths
    data_root = Path(args.data_root).resolve()
    runs_dir = Path(args.runs_dir).resolve()
    
    print("=" * 70)
    print("YOLO CLASSIFICATION TRAINING")
    print("=" * 70)
    print(f"Data root: {data_root}")
    print(f"Runs directory: {runs_dir}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device if args.device else 'auto'}")
    print(f"Seed: {args.seed}")
    print()
    
    # Validate data structure
    classify_dir = validate_data_structure(data_root)
    print(f"✓ Data structure validated: {classify_dir}")
    print()
    
    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    print(f"✓ Model loaded")
    print()
    
    # Set project directory
    project = args.project if args.project else str(runs_dir / "classify")
    
    # Training configuration
    train_config = {
        "data": str(classify_dir),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "patience": args.patience,
        "weight_decay": args.weight_decay,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "device": args.device,
        "seed": args.seed,
        "workers": args.workers,
        "project": project,
        "name": args.name,
        "exist_ok": args.exist_ok,
        "pretrained": args.pretrained,
        "optimizer": args.optimizer,
        "cos_lr": args.cos_lr,
        "dropout": args.dropout,
        "save": True,
        "save_period": -1,  # Save only best and last
        "plots": True,
        "verbose": True,
    }
    
    print("Training configuration:")
    print(json.dumps(train_config, indent=2))
    print()
    print("=" * 70)
    print("Starting training...")
    print("=" * 70)
    print()
    
    # Train model
    results = model.train(**train_config)
    
    print()
    print("=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"Results saved to: {results.save_dir}")
    print()
    
    # Print final metrics
    if hasattr(results, "results_dict"):
        print("Final metrics:")
        for key, value in results.results_dict.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
    
    # Save training arguments
    args_file = Path(results.save_dir) / "train_args.json"
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\n✓ Training arguments saved to: {args_file}")
    
    # Print best model path
    best_model = Path(results.save_dir) / "weights" / "best.pt"
    if best_model.exists():
        print(f"✓ Best model: {best_model}")
    
    last_model = Path(results.save_dir) / "weights" / "last.pt"
    if last_model.exists():
        print(f"✓ Last model: {last_model}")


if __name__ == "__main__":
    main()
