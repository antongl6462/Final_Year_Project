from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from evaluate_segmentation import build_semantic_model, run_full_evaluation, sliding_window_inference
from segmentation_dataset import (
    PatchSegmentationDataset,
    SegmentationSample,
    build_data_quality_report,
    build_train_augmentations,
    compute_positive_class_weight,
    detect_device,
    format_data_quality_report,
    package_versions,
    pair_kaggle_segmentation_samples,
    read_binary_mask,
    read_rgb_image,
    sample_background_only_images,
    sample_sanity_examples,
    set_global_seed,
    split_samples_no_leakage,
)
from utils_metrics import CombinedSegmentationLoss, compute_binary_metrics
from utils_visualisation import save_mask_sanity_grid, save_training_curves


@dataclass
class SegmentationConfig:
    dataset_root: str
    negative_dir: str
    project_root: str = "."
    architecture: str = "unet"
    encoder_name: str = "resnet34"
    patch_size: int = 512
    inference_overlap: float = 0.5
    positive_patch_prob: float = 0.7
    min_crack_pixels: int = 25
    patches_per_image: int = 8
    negative_ratio: float = 0.5
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    epochs: int = 100
    max_epochs: int = 200
    early_stopping_patience: int = 25
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 0
    accumulation_steps: int = 1
    num_workers: int = 0
    seed: int = 42
    mixed_precision: bool = True
    pretrained_encoder: bool = True
    recall_target: float = 0.85
    post_min_component_area: int = 0
    post_closing_kernel: int = 0
    post_dilation_kernel: int = 0
    max_train_steps: Optional[int] = None
    max_val_images: Optional[int] = None
    run_evaluation_after_training: bool = True


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_score = float("-inf")
        self.bad_epochs = 0

    def step(self, score: float) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def auto_batch_size(device: torch.device) -> int:
    if device.type == "cuda":
        return 4
    if device.type == "mps":
        return 2
    return 1


def ensure_output_dirs(project_root: Path) -> Dict[str, Path]:
    models_dir = project_root / "models"
    outputs_dir = project_root / "outputs"
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return {"models": models_dir, "outputs": outputs_dir}


def build_sample_inventory(config: SegmentationConfig) -> Dict[str, Sequence[SegmentationSample]]:
    positives = pair_kaggle_segmentation_samples(Path(config.dataset_root))
    background_count = int(len(positives) * config.negative_ratio)
    negatives = sample_background_only_images(Path(config.negative_dir), background_count, config.seed) if config.negative_dir else []
    all_samples = positives + negatives
    splits = split_samples_no_leakage(all_samples, config.train_ratio, config.val_ratio, config.test_ratio, config.seed)
    return {"positives": positives, "negatives": negatives, **splits}


def make_sanity_visuals(samples: Sequence[SegmentationSample], output_path: Path, seed: int) -> None:
    visual_items: List[Dict[str, Any]] = []
    for sample in sample_sanity_examples(samples, count=min(4, len(samples)), seed=seed):
        image = read_rgb_image(sample.image())
        raw_mask = cv2.imread(str(sample.mask()), cv2.IMREAD_GRAYSCALE) if sample.mask() else np.zeros(image.shape[:2], dtype=np.uint8)
        binary_mask = read_binary_mask(sample.mask(), image.shape[:2])
        overlay = image.copy()
        overlay[binary_mask > 0] = (
            overlay[binary_mask > 0] * 0.35 + np.array([255, 0, 0]) * 0.65
        ).astype(np.uint8)
        visual_items.append(
            {
                "stem": sample.stem,
                "image": image,
                "raw_mask": raw_mask,
                "binary_mask": binary_mask,
                "overlay": overlay,
            }
        )
    save_mask_sanity_grid(visual_items, output_path)


def evaluate_validation_epoch(
    model: torch.nn.Module,
    samples: Sequence[SegmentationSample],
    device: torch.device,
    config: SegmentationConfig,
) -> Dict[str, float]:
    probability_metrics: List[Dict[str, float]] = []
    max_images = config.max_val_images if config.max_val_images is not None else len(samples)
    was_training = model.training
    model.eval()
    try:
        for sample in list(samples)[:max_images]:
            image = read_rgb_image(sample.image())
            gt_mask = read_binary_mask(sample.mask(), image.shape[:2])
            probability_map = sliding_window_inference(
                model=model,
                image_rgb=image,
                device=device,
                patch_size=config.patch_size,
                overlap=config.inference_overlap,
                amp_enabled=bool(config.mixed_precision and device.type == "cuda"),
            )
            pred_mask = (probability_map >= 0.5).astype("uint8")
            probability_metrics.append(compute_binary_metrics(pred_mask, gt_mask))
    finally:
        if was_training:
            model.train()
    if not probability_metrics:
        return {key: 0.0 for key in ["val_dice", "val_iou", "val_precision", "val_recall", "val_f1"]}
    return {
        "val_dice": float(sum(row["dice"] for row in probability_metrics) / len(probability_metrics)),
        "val_iou": float(sum(row["iou"] for row in probability_metrics) / len(probability_metrics)),
        "val_precision": float(sum(row["precision"] for row in probability_metrics) / len(probability_metrics)),
        "val_recall": float(sum(row["recall"] for row in probability_metrics) / len(probability_metrics)),
        "val_f1": float(sum(row["f1"] for row in probability_metrics) / len(probability_metrics)),
    }


def save_checkpoint(path: Path, model: torch.nn.Module, config: SegmentationConfig, model_meta: Dict[str, Any], extra: Dict[str, Any]) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "model_meta": model_meta,
            **extra,
        },
        path,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: CombinedSegmentationLoss,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    accumulation_steps: int,
    mixed_precision: bool,
    max_train_steps: Optional[int],
) -> float:
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        with torch.autocast(device_type=autocast_device, enabled=bool(mixed_precision and device.type == "cuda")):
            logits = model(images)
            loss = loss_fn(logits, masks) / accumulation_steps
        if scaler is not None:
            scaler.scale(loss).backward()
            if step % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if step % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        running_loss += float(loss.item() * accumulation_steps)
        if max_train_steps is not None and step >= max_train_steps:
            break
    return running_loss / max(1, step)


def run_training_pipeline(config: SegmentationConfig) -> Dict[str, Any]:
    project_root = Path(config.project_root).resolve()
    output_dirs = ensure_output_dirs(project_root)
    set_global_seed(config.seed)
    device, device_name, amp_supported = detect_device()
    if config.batch_size <= 0:
        config.batch_size = auto_batch_size(device)
    if config.accumulation_steps <= 0:
        config.accumulation_steps = 1

    inventory = build_sample_inventory(config)
    report = build_data_quality_report(inventory["positives"] + inventory["negatives"])
    (output_dirs["outputs"] / "data_quality_report.json").write_text(format_data_quality_report(report) + "\n")
    make_sanity_visuals(inventory["train"], output_dirs["outputs"] / "mask_sanity_check.png", config.seed)

    train_transform = build_train_augmentations()
    train_dataset = PatchSegmentationDataset(
        inventory["train"],
        patch_size=config.patch_size,
        positive_patch_prob=config.positive_patch_prob,
        min_crack_pixels=config.min_crack_pixels,
        patches_per_image=config.patches_per_image,
        transform=train_transform,
        seed=config.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    pos_weight = compute_positive_class_weight(inventory["train"])
    model, model_meta = build_semantic_model(
        architecture=config.architecture,
        encoder_name=config.encoder_name,
        pretrained=config.pretrained_encoder,
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    loss_fn = CombinedSegmentationLoss(pos_weight=pos_weight).to(device)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" and config.mixed_precision else None
    stopper = EarlyStopping(config.early_stopping_patience)

    history: List[Dict[str, Any]] = []
    best_dice = float("-inf")
    best_path = output_dirs["models"] / "best_semantic_segmentation.pth"
    last_path = output_dirs["models"] / "last_semantic_segmentation.pth"

    start_time = time.time()
    print("Semantic segmentation environment")
    print(json.dumps({"device": device_name, "amp_enabled": bool(amp_supported and config.mixed_precision), "versions": package_versions()}, indent=2))
    print("Data quality report")
    print(format_data_quality_report(report))
    if device.type == "cpu":
        print("Warning: CPU training will be very slow for full 100+ epoch runs.")

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            accumulation_steps=config.accumulation_steps,
            mixed_precision=config.mixed_precision,
            max_train_steps=config.max_train_steps,
        )
        scheduler.step()
        val_metrics = evaluate_validation_epoch(model, inventory["val"], device, config)
        epoch_row = {"epoch": epoch, "train_loss": train_loss, **val_metrics, "lr": optimizer.param_groups[0]["lr"]}
        history.append(epoch_row)
        print(json.dumps(epoch_row, indent=2))

        if val_metrics["val_dice"] > best_dice:
            best_dice = val_metrics["val_dice"]
            save_checkpoint(best_path, model, config, model_meta, {"best_val_dice": best_dice, "history": history})
        save_checkpoint(last_path, model, config, model_meta, {"best_val_dice": best_dice, "history": history})

        if stopper.step(val_metrics["val_dice"]):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dirs["outputs"] / "training_history.csv", index=False)
    save_training_curves(history_df, output_dirs["outputs"] / "training_curves.png")

    evaluation_results = None
    if config.run_evaluation_after_training and best_path.exists():
        evaluation_results = run_full_evaluation(
            checkpoint_path=best_path,
            val_samples=inventory["val"],
            test_samples=inventory["test"],
            config=asdict(config),
            output_dir=output_dirs["outputs"],
            device=device,
        )

    return {
        "history": history_df,
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "outputs_dir": output_dirs["outputs"],
        "models_dir": output_dirs["models"],
        "train_minutes": (time.time() - start_time) / 60.0,
        "evaluation": evaluation_results,
        "data_quality_report": report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train raster semantic segmentation for concrete cracks")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--negative_dir", type=str, default="")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--architecture", type=str, default="unet")
    parser.add_argument("--encoder_name", type=str, default="resnet34")
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--inference_overlap", type=float, default=0.5)
    parser.add_argument("--positive_patch_prob", type=float, default=0.7)
    parser.add_argument("--min_crack_pixels", type=int, default=25)
    parser.add_argument("--patches_per_image", type=int, default=8)
    parser.add_argument("--negative_ratio", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--early_stopping_patience", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_val_images", type=int, default=None)
    parser.add_argument("--skip_evaluation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_kwargs = vars(args).copy()
    skip_evaluation = bool(config_kwargs.pop("skip_evaluation"))
    config_kwargs["run_evaluation_after_training"] = not skip_evaluation
    config = SegmentationConfig(**config_kwargs)
    results = run_training_pipeline(config)
    print(f"Best checkpoint: {results['best_checkpoint']}")
    print(f"Last checkpoint: {results['last_checkpoint']}")
    if results["evaluation"] is not None:
        print(results["evaluation"]["summary_text"])


if __name__ == "__main__":
    main()
