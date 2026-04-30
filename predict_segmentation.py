from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluate_segmentation import load_checkpoint, sliding_window_inference
from segmentation_dataset import detect_device, read_rgb_image
from utils_metrics import PostProcessingConfig, apply_post_processing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict crack masks with the semantic segmentation checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Image file or folder")
    parser.add_argument("--output_dir", type=str, default="outputs/predictions")
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min_component_area", type=int, default=0)
    parser.add_argument("--closing_kernel", type=int, default=0)
    parser.add_argument("--dilation_kernel", type=int, default=0)
    return parser.parse_args()


def iter_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([path for path in input_path.iterdir() if path.suffix.lower() in extensions])


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_rgb.copy()
    overlay[mask > 0] = (overlay[mask > 0] * 0.35 + np.array([255, 0, 0]) * 0.65).astype(np.uint8)
    return overlay


def main() -> None:
    args = parse_args()
    device = detect_device()[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, _ = load_checkpoint(Path(args.checkpoint), device)
    post_cfg = PostProcessingConfig(
        min_component_area=args.min_component_area,
        closing_kernel=args.closing_kernel,
        dilation_kernel=args.dilation_kernel,
    )

    for image_path in iter_inputs(Path(args.input)):
        image_rgb = read_rgb_image(image_path)
        probability_map = sliding_window_inference(
            model=model,
            image_rgb=image_rgb,
            device=device,
            patch_size=args.patch_size,
            overlap=args.overlap,
            amp_enabled=device.type == "cuda",
        )
        pred_mask = (probability_map >= args.threshold).astype(np.uint8)
        pred_mask = apply_post_processing(pred_mask, post_cfg)
        overlay = overlay_mask(image_rgb, pred_mask)
        stem = image_path.stem
        cv2.imwrite(str(output_dir / f"{stem}_probability.png"), (probability_map * 255).astype(np.uint8))
        cv2.imwrite(str(output_dir / f"{stem}_mask.png"), (pred_mask * 255).astype(np.uint8))
        cv2.imwrite(str(output_dir / f"{stem}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved prediction outputs for {image_path.name}")


if __name__ == "__main__":
    main()
