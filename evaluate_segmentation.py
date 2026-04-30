from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

from segmentation_dataset import (
    build_eval_augmentations,
    detect_device,
    pair_kaggle_segmentation_samples,
    prepare_inference_patch,
    read_binary_mask,
    read_rgb_image,
    sample_background_only_images,
    sliding_window_coordinates,
    split_samples_no_leakage,
)
from utils_metrics import (
    PostProcessingConfig,
    aggregate_confusion_metrics,
    apply_post_processing,
    compute_binary_metrics,
    select_operating_threshold,
    threshold_sweep,
)
from utils_visualisation import (
    save_confusion_matrix,
    save_qualitative_predictions,
    save_threshold_curves,
    save_worst_predictions,
)

try:
    smp = importlib.import_module("segmentation_models_pytorch")
except Exception:
    smp = None


class TorchvisionDeepLabBinary(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        if pretrained:
            try:
                model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
            except Exception:
                model = deeplabv3_resnet50(weights=None, weights_backbone=None)
        else:
            model = deeplabv3_resnet50(weights=None, weights_backbone=None)
        model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        if getattr(model, "aux_classifier", None) is not None:
            model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


def build_semantic_model(
    architecture: str = "unet",
    encoder_name: str = "resnet34",
    pretrained: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    architecture = architecture.lower()
    metadata = {"architecture": architecture, "encoder_name": encoder_name}

    if smp is not None:
        encoder_weights = "imagenet" if pretrained else None
        try:
            if architecture == "unetplusplus":
                model = smp.UnetPlusPlus(
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=3,
                    classes=1,
                    activation=None,
                )
            elif architecture == "deeplabv3plus":
                model = smp.DeepLabV3Plus(
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=3,
                    classes=1,
                    activation=None,
                )
            else:
                model = smp.Unet(
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=3,
                    classes=1,
                    activation=None,
                )
            metadata["framework"] = "segmentation_models_pytorch"
            return model, metadata
        except Exception as exc:
            metadata["fallback_warning"] = str(exc)

    metadata["framework"] = "torchvision"
    metadata["architecture"] = "deeplabv3"
    metadata["encoder_name"] = "resnet50"
    return TorchvisionDeepLabBinary(pretrained=pretrained), metadata


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_meta = checkpoint.get("model_meta", {})
    model, created_meta = build_semantic_model(
        architecture=model_meta.get("architecture", "unet"),
        encoder_name=model_meta.get("encoder_name", "resnet34"),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    checkpoint["model_meta"] = {**created_meta, **model_meta}
    return model, checkpoint


@torch.no_grad()
def sliding_window_inference(
    model: nn.Module,
    image_rgb: np.ndarray,
    device: torch.device,
    patch_size: int = 512,
    overlap: float = 0.5,
    amp_enabled: bool = False,
) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    coords = sliding_window_coordinates(height, width, patch_size, overlap)
    eval_transform = build_eval_augmentations()
    prob_sum = np.zeros((height, width), dtype=np.float32)
    count_sum = np.zeros((height, width), dtype=np.float32)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    for top, left, bottom, right in coords:
        patch = image_rgb[top:bottom, left:right]
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            padded = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
            padded[: patch.shape[0], : patch.shape[1]] = patch
            patch = padded

        patch_tensor = prepare_inference_patch(patch, eval_transform=eval_transform).unsqueeze(0).to(device)
        with torch.autocast(device_type=autocast_device, enabled=amp_enabled):
            logits = model(patch_tensor)
            probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

        prob_sum[top:bottom, left:right] += probs[: bottom - top, : right - left]
        count_sum[top:bottom, left:right] += 1.0

    return prob_sum / np.maximum(count_sum, 1.0)


def collect_probability_maps(
    model: nn.Module,
    samples: Sequence,
    device: torch.device,
    patch_size: int,
    overlap: float,
    amp_enabled: bool,
    max_images: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    probability_maps: List[np.ndarray] = []
    gt_masks: List[np.ndarray] = []
    for index, sample in enumerate(samples):
        if max_images is not None and index >= max_images:
            break
        image = read_rgb_image(sample.image())
        gt_mask = read_binary_mask(sample.mask(), image_shape=image.shape[:2])
        probability_map = sliding_window_inference(
            model=model,
            image_rgb=image,
            device=device,
            patch_size=patch_size,
            overlap=overlap,
            amp_enabled=amp_enabled,
        )
        probability_maps.append(probability_map)
        gt_masks.append(gt_mask)
    return probability_maps, gt_masks


def evaluate_samples_with_threshold(
    samples: Sequence,
    probability_maps: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    threshold: float,
    postprocessing: Optional[PostProcessingConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, float], np.ndarray, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    qualitative_items: List[Dict[str, Any]] = []
    confusion = np.zeros((2, 2), dtype=np.int64)
    for sample, probability_map, gt_mask in zip(samples, probability_maps, gt_masks):
        pred_mask = (probability_map >= threshold).astype(np.uint8)
        pred_mask = apply_post_processing(pred_mask, postprocessing)
        metrics = compute_binary_metrics(pred_mask, gt_mask)
        rows.append({"stem": sample.stem, "source": sample.source, **metrics})
        qualitative_items.append(
            {
                "stem": sample.stem,
                "image": read_rgb_image(sample.image()),
                "gt_mask": gt_mask,
                "probability_map": probability_map,
                "pred_mask": pred_mask,
                **metrics,
            }
        )
        confusion[0, 0] += int(metrics["tn"])
        confusion[0, 1] += int(metrics["fp"])
        confusion[1, 0] += int(metrics["fn"])
        confusion[1, 1] += int(metrics["tp"])

    per_image_df = pd.DataFrame(rows).sort_values("dice").reset_index(drop=True)
    summary = aggregate_confusion_metrics(rows)
    return per_image_df, summary, confusion, qualitative_items


def dissertation_summary_text(summary: Dict[str, float], threshold: float, recall_priority: bool) -> str:
    operating_mode = "recall-prioritised" if recall_priority else "F1-optimal"
    return (
        "Original YOLO segmentation failed mainly because polygon conversion removed thin-crack geometry before training, "
        "so the supervision itself was degraded. The revised pipeline uses raster-mask semantic segmentation throughout, "
        "which preserves thin crack structure at pixel level. "
        f"Final test Dice={summary['dice']:.4f}, IoU={summary['iou']:.4f}, precision={summary['precision']:.4f}, "
        f"recall={summary['recall']:.4f}. "
        f"The selected threshold is {threshold:.2f} using the {operating_mode} operating point. "
        "Recall is prioritised because missing structural cracks is more costly than accepting extra false positives."
    )


def run_full_evaluation(
    checkpoint_path: Path,
    val_samples: Sequence,
    test_samples: Sequence,
    config: Dict[str, Any],
    output_dir: Path,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    device = device or detect_device()[0]
    amp_enabled = bool(config.get("mixed_precision", False) and device.type == "cuda")
    patch_size = int(config.get("patch_size", 512))
    overlap = float(config.get("inference_overlap", 0.5))
    recall_target = float(config.get("recall_target", 0.85))
    output_dir.mkdir(parents=True, exist_ok=True)

    post_cfg = PostProcessingConfig(
        min_component_area=int(config.get("post_min_component_area", 0)),
        closing_kernel=int(config.get("post_closing_kernel", 0)),
        dilation_kernel=int(config.get("post_dilation_kernel", 0)),
    )

    model, checkpoint = load_checkpoint(checkpoint_path, device)

    val_probability_maps, val_gt_masks = collect_probability_maps(
        model=model,
        samples=val_samples,
        device=device,
        patch_size=patch_size,
        overlap=overlap,
        amp_enabled=amp_enabled,
        max_images=config.get("max_val_images"),
    )
    threshold_df = threshold_sweep(val_probability_maps, val_gt_masks, np.linspace(0.05, 0.95, 19))
    best_f1_threshold, recall_threshold = select_operating_threshold(threshold_df, recall_target=recall_target)
    threshold_df.to_csv(output_dir / "threshold_sweep.csv", index=False)
    save_threshold_curves(threshold_df, output_dir / "threshold_curves.png")

    chosen_threshold = recall_threshold
    test_probability_maps, test_gt_masks = collect_probability_maps(
        model=model,
        samples=test_samples,
        device=device,
        patch_size=patch_size,
        overlap=overlap,
        amp_enabled=amp_enabled,
    )

    before_df, before_summary, _, _ = evaluate_samples_with_threshold(
        samples=test_samples,
        probability_maps=test_probability_maps,
        gt_masks=test_gt_masks,
        threshold=chosen_threshold,
        postprocessing=None,
    )
    after_df, after_summary, confusion, qualitative_items = evaluate_samples_with_threshold(
        samples=test_samples,
        probability_maps=test_probability_maps,
        gt_masks=test_gt_masks,
        threshold=chosen_threshold,
        postprocessing=post_cfg,
    )

    pd.DataFrame(
        [
            {"mode": "before_postprocessing", **before_summary},
            {"mode": "after_postprocessing", **after_summary},
        ]
    ).to_csv(output_dir / "segmentation_metrics_test.csv", index=False)
    after_df.to_csv(output_dir / "per_image_metrics_test.csv", index=False)
    save_confusion_matrix(confusion, output_dir / "confusion_matrix.png")
    save_qualitative_predictions(
        qualitative_items,
        output_dir / "qualitative_predictions.png",
        "Semantic Segmentation Predictions",
    )
    save_worst_predictions(qualitative_items, output_dir / "worst_predictions.png")

    summary_text = dissertation_summary_text(after_summary, chosen_threshold, recall_priority=True)
    (output_dir / "dissertation_summary.txt").write_text(summary_text + "\n")
    return {
        "threshold_df": threshold_df,
        "best_f1_threshold": best_f1_threshold,
        "chosen_threshold": chosen_threshold,
        "before_df": before_df,
        "after_df": after_df,
        "summary": after_summary,
        "summary_text": summary_text,
        "checkpoint": checkpoint,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the raster semantic segmentation baseline")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--negative_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--negative_ratio", type=float, default=0.5)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--inference_overlap", type=float, default=0.5)
    parser.add_argument("--recall_target", type=float, default=0.85)
    parser.add_argument("--post_min_component_area", type=int, default=0)
    parser.add_argument("--post_closing_kernel", type=int, default=0)
    parser.add_argument("--post_dilation_kernel", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positives = pair_kaggle_segmentation_samples(Path(args.dataset_root))
    negatives = sample_background_only_images(
        Path(args.negative_dir),
        target_count=int(len(positives) * args.negative_ratio),
        seed=args.seed,
    ) if args.negative_dir else []
    splits = split_samples_no_leakage(positives + negatives, 0.8, 0.1, 0.1, args.seed)
    evaluation = run_full_evaluation(
        checkpoint_path=Path(args.checkpoint),
        val_samples=splits["val"],
        test_samples=splits["test"],
        config=vars(args),
        output_dir=Path(args.output_dir),
    )
    print(evaluation["summary_text"])


if __name__ == "__main__":
    main()
