#!/usr/bin/env python3
"""
Train YOLO segmentation model for concrete crack detection.

Usage:
    python src/train_yolo_segment.py --data_yaml /path/to/data/segment_yolo/crack-seg.yaml --runs_dir ./runs
"""

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO segmentation model")
    parser.add_argument(
        "--data_yaml",
        type=str,
        required=True,
        help="Path to YOLO dataset YAML file (e.g., DATA_ROOT/segment_yolo/crack-seg.yaml)",
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
        default="yolov8n-seg.pt",
        help="YOLO segmentation model (default: yolov8n-seg.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Maximum number of training epochs (default: 150)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=256,
        help="Image size for training (default: 256)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience (default: 30)",
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
        help="Project name (default: RUNS_DIR/segment)",
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
        "--close_mosaic",
        type=int,
        default=10,
        help="Epochs to disable mosaic augmentation before end (default: 10)",
    )
    parser.add_argument(
        "--overlap_mask",
        action="store_true",
        default=True,
        help="Allow mask overlap during training (default: True)",
    )
    parser.add_argument(
        "--mask_ratio",
        type=int,
        default=4,
        help="Mask downsample ratio (default: 4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Resolve paths
    data_yaml = Path(args.data_yaml).resolve()
    runs_dir = Path(args.runs_dir).resolve()
    
    if not data_yaml.exists():
        raise ValueError(
            f"Dataset YAML not found: {data_yaml}\n"
            "Please run prepare_segment_data.py first."
        )
    
    print("=" * 70)
    print("YOLO SEGMENTATION TRAINING")
    print("=" * 70)
    print(f"Data YAML: {data_yaml}")
    print(f"Runs directory: {runs_dir}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device if args.device else 'auto'}")
    print(f"Seed: {args.seed}")
    print()
    
    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    print(f"✓ Model loaded")
    print()
    
    # Set project directory
    project = args.project if args.project else str(runs_dir / "segment")
    
    # Training configuration
    train_config = {
        "data": str(data_yaml),
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
        "close_mosaic": args.close_mosaic,
        "overlap_mask": args.overlap_mask,
        "mask_ratio": args.mask_ratio,
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
