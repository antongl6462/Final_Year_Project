from __future__ import annotations

import importlib.metadata
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

try:
	import albumentations as A
except Exception:
	A = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SegmentationSample:
	image_path: str
	mask_path: Optional[str]
	stem: str
	group_id: str
	source: str
	is_background_only: bool = False

	def image(self) -> Path:
		return Path(self.image_path)

	def mask(self) -> Optional[Path]:
		return Path(self.mask_path) if self.mask_path else None


def safe_version(package_name: str) -> str:
	try:
		return importlib.metadata.version(package_name)
	except importlib.metadata.PackageNotFoundError:
		return "not-installed"


def package_versions() -> Dict[str, str]:
	return {
		"python": f"{torch.__version__ if False else ''}".strip() or __import__("sys").version.split()[0],
		"torch": safe_version("torch"),
		"torchvision": safe_version("torchvision"),
		"opencv-python": safe_version("opencv-python"),
		"numpy": safe_version("numpy"),
		"albumentations": safe_version("albumentations"),
		"segmentation-models-pytorch": safe_version("segmentation-models-pytorch"),
		"pandas": safe_version("pandas"),
		"scikit-image": safe_version("scikit-image"),
	}


def set_global_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	try:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	except Exception:
		pass


def detect_device() -> Tuple[torch.device, str, bool]:
	if torch.cuda.is_available():
		return torch.device("cuda"), torch.cuda.get_device_name(0), True
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return torch.device("mps"), "Apple Metal (MPS)", False
	return torch.device("cpu"), "CPU", False


def _iter_image_files(directory: Path) -> List[Path]:
	extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
	return sorted([path for path in directory.iterdir() if path.suffix.lower() in extensions])


def read_rgb_image(image_path: Path) -> np.ndarray:
	image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
	if image_bgr is None:
		raise FileNotFoundError(f"Could not read image: {image_path}")
	return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def read_binary_mask(mask_path: Optional[Path], image_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
	if mask_path is None:
		if image_shape is None:
			raise ValueError("image_shape is required when mask_path is None")
		return np.zeros(image_shape, dtype=np.uint8)

	mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
	if mask is None:
		raise FileNotFoundError(f"Could not read mask: {mask_path}")

	if image_shape is not None and mask.shape[:2] != image_shape:
		mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)

	return (mask > 0).astype(np.uint8)


def resolve_kaggle_segmentation_root(dataset_root: Path) -> Tuple[Path, Path]:
	dataset_root = dataset_root.resolve()
	candidates = [
		dataset_root,
		dataset_root / "concreteCrackSegmentationDataset",
	]
	for candidate in candidates:
		rgb_dir = candidate / "rgb"
		bw_dir = candidate / "BW"
		if rgb_dir.exists() and bw_dir.exists():
			return rgb_dir, bw_dir
	raise FileNotFoundError(
		f"Could not find Kaggle segmentation folders under {dataset_root}. "
		"Expected `rgb/` and `BW/` folders."
	)


def pair_kaggle_segmentation_samples(dataset_root: Path) -> List[SegmentationSample]:
	rgb_dir, bw_dir = resolve_kaggle_segmentation_root(dataset_root)
	rgb_map = {path.stem: path for path in _iter_image_files(rgb_dir)}
	bw_map = {path.stem: path for path in _iter_image_files(bw_dir)}
	shared_stems = sorted(set(rgb_map) & set(bw_map))
	if not shared_stems:
		raise RuntimeError("No shared image/mask stems found in Kaggle segmentation dataset")

	return [
		SegmentationSample(
			image_path=str(rgb_map[stem]),
			mask_path=str(bw_map[stem]),
			stem=stem,
			group_id=f"positive::{stem}",
			source="kaggle_positive",
			is_background_only=False,
		)
		for stem in shared_stems
	]


def sample_background_only_images(
	negative_dir: Path,
	target_count: int,
	seed: int,
) -> List[SegmentationSample]:
	if not negative_dir.exists():
		return []
	negative_images = _iter_image_files(negative_dir)
	if not negative_images:
		return []
	chosen = negative_images if target_count >= len(negative_images) else random.Random(seed).sample(negative_images, target_count)
	return [
		SegmentationSample(
			image_path=str(path),
			mask_path=None,
			stem=path.stem,
			group_id=f"negative::{path.stem}",
			source="classification_negative",
			is_background_only=True,
		)
		for path in chosen
	]


def maybe_inverted_mask_warning(crack_ratios: Sequence[float]) -> Optional[str]:
	if not crack_ratios:
		return None
	mean_ratio = float(np.mean(crack_ratios))
	if mean_ratio > 0.5:
		return "Masks may be inverted: mean foreground ratio exceeds 50%."
	return None


def build_data_quality_report(samples: Sequence[SegmentationSample]) -> Dict[str, Any]:
	report: Dict[str, Any] = {
		"num_pairs": 0,
		"num_background_only": 0,
		"num_total_samples": len(samples),
		"num_empty_masks": 0,
		"num_size_mismatches": 0,
		"resolution_counts": Counter(),
		"crack_pixel_ratio_min": 0.0,
		"crack_pixel_ratio_mean": 0.0,
		"crack_pixel_ratio_max": 0.0,
		"crack_pixel_ratios": [],
		"warnings": [],
	}

	crack_ratios: List[float] = []
	for sample in samples:
		image = read_rgb_image(sample.image())
		image_shape = image.shape[:2]
		mask_path = sample.mask()
		if mask_path is not None:
			raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
			if raw_mask is None:
				raise FileNotFoundError(f"Could not read mask: {mask_path}")
			if raw_mask.shape[:2] != image_shape:
				report["num_size_mismatches"] += 1
			mask = read_binary_mask(mask_path, image_shape=image_shape)
			report["num_pairs"] += 1
		else:
			mask = np.zeros(image_shape, dtype=np.uint8)
			report["num_background_only"] += 1

		crack_ratio = float(mask.mean())
		crack_ratios.append(crack_ratio)
		if mask.sum() == 0:
			report["num_empty_masks"] += 1
		report["resolution_counts"][f"{image_shape[1]}x{image_shape[0]}"] += 1

	if crack_ratios:
		report["crack_pixel_ratios"] = crack_ratios
		report["crack_pixel_ratio_min"] = float(np.min(crack_ratios))
		report["crack_pixel_ratio_mean"] = float(np.mean(crack_ratios))
		report["crack_pixel_ratio_max"] = float(np.max(crack_ratios))

	inversion_warning = maybe_inverted_mask_warning(crack_ratios)
	if inversion_warning:
		report["warnings"].append(inversion_warning)
	if report["num_size_mismatches"] > 0:
		report["warnings"].append("Some image/mask pairs have mismatched sizes.")
	return report


def format_data_quality_report(report: Dict[str, Any]) -> str:
	summary = {
		key: value
		for key, value in report.items()
		if key not in {"resolution_counts", "crack_pixel_ratios"}
	}
	summary["resolution_counts"] = dict(report["resolution_counts"])
	return json.dumps(summary, indent=2)


def sample_sanity_examples(samples: Sequence[SegmentationSample], count: int, seed: int) -> List[SegmentationSample]:
	rng = random.Random(seed)
	usable = list(samples)
	return usable if len(usable) <= count else rng.sample(usable, count)


def split_samples_no_leakage(
	samples: Sequence[SegmentationSample],
	train_ratio: float,
	val_ratio: float,
	test_ratio: float,
	seed: int,
) -> Dict[str, List[SegmentationSample]]:
	if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
		raise ValueError("Split ratios must sum to 1.0")

	groups = list(samples)
	labels = [0 if sample.is_background_only else 1 for sample in groups]
	indices = list(range(len(groups)))

	train_idx, temp_idx = train_test_split(
		indices,
		train_size=train_ratio,
		random_state=seed,
		stratify=labels,
	)
	temp_labels = [labels[index] for index in temp_idx]
	val_fraction = val_ratio / (val_ratio + test_ratio)
	val_idx, test_idx = train_test_split(
		temp_idx,
		train_size=val_fraction,
		random_state=seed,
		stratify=temp_labels,
	)

	return {
		"train": [groups[index] for index in train_idx],
		"val": [groups[index] for index in val_idx],
		"test": [groups[index] for index in test_idx],
	}


def compute_positive_class_weight(samples: Sequence[SegmentationSample], clamp: Tuple[float, float] = (1.0, 20.0)) -> float:
	positive_pixels = 0
	total_pixels = 0
	for sample in samples:
		image = read_rgb_image(sample.image())
		mask = read_binary_mask(sample.mask(), image_shape=image.shape[:2])
		positive_pixels += int(mask.sum())
		total_pixels += int(mask.size)
	negative_pixels = max(total_pixels - positive_pixels, 1)
	positive_pixels = max(positive_pixels, 1)
	pos_weight = negative_pixels / positive_pixels
	return float(np.clip(pos_weight, clamp[0], clamp[1]))


def pad_image_and_mask(
	image: np.ndarray,
	mask: np.ndarray,
	patch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
	height, width = image.shape[:2]
	pad_h = max(0, patch_size - height)
	pad_w = max(0, patch_size - width)
	if pad_h == 0 and pad_w == 0:
		return image, mask
	image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
	mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
	return image, mask


def _manual_normalize(image: np.ndarray) -> np.ndarray:
	image = image.astype(np.float32) / 255.0
	mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
	std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
	return (image - mean) / std


def build_train_augmentations() -> Any:
	if A is None:
		return None
	return A.Compose(
		[
			A.HorizontalFlip(p=0.5),
			A.VerticalFlip(p=0.5),
			A.RandomRotate90(p=0.5),
			A.ShiftScaleRotate(
				shift_limit=0.05,
				scale_limit=0.1,
				rotate_limit=15,
				interpolation=cv2.INTER_LINEAR,
				border_mode=cv2.BORDER_REFLECT_101,
				p=0.6,
			),
			A.RandomBrightnessContrast(p=0.4),
			A.CLAHE(p=0.25),
			A.GaussNoise(std_range=(0.01, 0.04), p=0.25),
			A.GaussianBlur(blur_limit=(3, 5), p=0.15),
			A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
		]
	)


def build_eval_augmentations() -> Any:
	if A is None:
		return None
	return A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])


class PatchSegmentationDataset(Dataset):
	def __init__(
		self,
		samples: Sequence[SegmentationSample],
		patch_size: int = 512,
		positive_patch_prob: float = 0.7,
		min_crack_pixels: int = 25,
		patches_per_image: int = 8,
		transform: Any = None,
		seed: int = 42,
	) -> None:
		self.samples = list(samples)
		self.patch_size = patch_size
		self.positive_patch_prob = positive_patch_prob
		self.min_crack_pixels = min_crack_pixels
		self.patches_per_image = patches_per_image
		self.transform = transform
		self.seed = seed
		self.positive_indices = [
			index for index, sample in enumerate(self.samples) if not sample.is_background_only
		]

	def __len__(self) -> int:
		return len(self.samples) * self.patches_per_image

	def _pick_patch_coordinates(self, mask: np.ndarray, rng: random.Random) -> Tuple[int, int]:
		height, width = mask.shape[:2]
		if mask.sum() >= self.min_crack_pixels and rng.random() < self.positive_patch_prob:
			positive_pixels = np.argwhere(mask > 0)
			if len(positive_pixels) > 0:
				center_y, center_x = positive_pixels[rng.randrange(len(positive_pixels))]
				top = int(np.clip(center_y - self.patch_size // 2 + rng.randint(-32, 32), 0, max(height - self.patch_size, 0)))
				left = int(np.clip(center_x - self.patch_size // 2 + rng.randint(-32, 32), 0, max(width - self.patch_size, 0)))
				return top, left
		top = rng.randint(0, max(height - self.patch_size, 0)) if height > self.patch_size else 0
		left = rng.randint(0, max(width - self.patch_size, 0)) if width > self.patch_size else 0
		return top, left

	def __getitem__(self, index: int) -> Dict[str, Any]:
		rng = random.Random(self.seed + index)
		sample = self.samples[index % len(self.samples)]
		image = read_rgb_image(sample.image())
		mask = read_binary_mask(sample.mask(), image_shape=image.shape[:2])
		image, mask = pad_image_and_mask(image, mask, self.patch_size)

		patch_image = None
		patch_mask = None
		for _ in range(12):
			top, left = self._pick_patch_coordinates(mask, rng)
			patch_image = image[top : top + self.patch_size, left : left + self.patch_size]
			patch_mask = mask[top : top + self.patch_size, left : left + self.patch_size]
			if patch_mask.sum() >= self.min_crack_pixels or sample.is_background_only or rng.random() > self.positive_patch_prob:
				break

		if patch_image is None or patch_mask is None:
			raise RuntimeError("Failed to sample segmentation patch")

		if self.transform is not None:
			transformed = self.transform(image=patch_image, mask=patch_mask)
			patch_image = transformed["image"]
			patch_mask = transformed["mask"]
		else:
			patch_image = _manual_normalize(patch_image)

		image_tensor = torch.from_numpy(np.transpose(patch_image, (2, 0, 1))).float()
		mask_tensor = torch.from_numpy(patch_mask[None, ...].astype(np.float32))
		return {
			"image": image_tensor,
			"mask": mask_tensor,
			"stem": sample.stem,
			"source": sample.source,
		}


def sliding_window_coordinates(
	height: int,
	width: int,
	patch_size: int,
	overlap: float,
) -> List[Tuple[int, int, int, int]]:
	stride = max(1, int(round(patch_size * (1.0 - overlap))))
	y_starts = list(range(0, max(height - patch_size, 0) + 1, stride))
	x_starts = list(range(0, max(width - patch_size, 0) + 1, stride))
	if not y_starts or y_starts[-1] != max(height - patch_size, 0):
		y_starts.append(max(height - patch_size, 0))
	if not x_starts or x_starts[-1] != max(width - patch_size, 0):
		x_starts.append(max(width - patch_size, 0))
	return [(top, left, min(top + patch_size, height), min(left + patch_size, width)) for top in y_starts for left in x_starts]


def prepare_inference_patch(image_patch: np.ndarray, eval_transform: Any = None) -> torch.Tensor:
	if eval_transform is not None:
		image_patch = eval_transform(image=image_patch)["image"]
	else:
		image_patch = _manual_normalize(image_patch)
	return torch.from_numpy(np.transpose(image_patch, (2, 0, 1))).float()


def mask_to_yolo_polygons_validated(
	mask: np.ndarray,
	min_contour_area: float = 5.0,
	epsilon_fraction: float = 0.0,
	required_iou: float = 0.98,
) -> List[str]:
	if mask.dtype != np.uint8:
		mask = mask.astype(np.uint8)
	mask = (mask > 0).astype(np.uint8)
	height, width = mask.shape[:2]
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	lines: List[str] = []
	reconstructed = np.zeros_like(mask)

	for contour in contours:
		area = cv2.contourArea(contour)
		if area < min_contour_area:
			continue
		if epsilon_fraction > 0.0:
			epsilon = epsilon_fraction * cv2.arcLength(contour, True)
			contour = cv2.approxPolyDP(contour, epsilon, True)
		points = contour.reshape(-1, 2).astype(np.float32)
		if len(points) < 3:
			continue
		normalized = points.copy()
		normalized[:, 0] = np.clip(normalized[:, 0] / width, 0.0, 1.0)
		normalized[:, 1] = np.clip(normalized[:, 1] / height, 0.0, 1.0)
		lines.append("0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in normalized))
		cv2.fillPoly(reconstructed, [points.astype(np.int32)], 1)

	intersection = np.logical_and(mask > 0, reconstructed > 0).sum()
	union = np.logical_or(mask > 0, reconstructed > 0).sum()
	iou = float(intersection / union) if union > 0 else 1.0
	if iou < required_iou:
		raise ValueError(
			f"YOLO polygon reconstruction IoU {iou:.4f} is below required {required_iou:.4f}."
		)
	return lines
