"""Data handling for hybrid QCNN experiments.

This module reuses the repository's existing splitting logic where possible and
falls back to a reproducible stratified split from raw data.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from prepare_classify_data import get_image_paths, stratified_split

from .config import QuantumCNNConfig
from .quantum_preprocessing import QuantumPreprocessConfig, preprocess_for_quantum


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


class PathLabelDataset(Dataset):
    """Simple dataset backed by `(path, label)` pairs."""

    def __init__(self, paths: Sequence[Path], labels: Sequence[int], transform=None):
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = Image.open(self.paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]


class QuantumTransform:
    """Composite transform with optional quantum-specific preprocessing."""

    def __init__(self, config: QuantumCNNConfig):
        self.enable_quantum_preprocess = config.quantum_preprocess
        self.preprocess_cfg = QuantumPreprocessConfig(
            use_grayscale=config.use_grayscale,
            use_clahe=config.use_clahe,
            use_blackhat=config.use_blackhat,
        )
        self.resize = transforms.Resize((config.image_size, config.image_size))
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if self.enable_quantum_preprocess:
            image = preprocess_for_quantum(image, self.preprocess_cfg)
        image = self.resize(image)
        tensor = self.to_tensor(image)
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor


def _list_split(split_root: Path, class_names: List[str]) -> Tuple[List[Path], List[int]]:
    """Collect `(path, label)` from an existing classify split directory."""
    split_paths: List[Path] = []
    split_labels: List[int] = []
    for label_index, class_name in enumerate(class_names):
        class_dir = split_root / class_name
        if not class_dir.exists():
            continue
        images = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
        split_paths.extend(images)
        split_labels.extend([label_index] * len(images))
    return split_paths, split_labels


def _cap_split(paths: List[Path], labels: List[int], max_samples: int | None, seed: int) -> Tuple[List[Path], List[int]]:
    """Subsample split for bounded runtime without changing label semantics."""
    if max_samples is None or len(paths) <= max_samples:
        return paths, labels

    indices = list(range(len(paths)))
    random.Random(seed).shuffle(indices)
    chosen = sorted(indices[:max_samples])
    return [paths[i] for i in chosen], [labels[i] for i in chosen]


def _load_from_existing_split(config: QuantumCNNConfig):
    """Load splits from `classify_yolo` if already prepared by classical workflow."""
    from pathlib import Path
    dataset_root = Path(config.dataset_root) if isinstance(config.dataset_root, str) else config.dataset_root
    classify_root = dataset_root.parent / "classify_yolo"
    if not classify_root.exists():
        return None

    train_split = classify_root / "train"
    class_dirs = sorted([d for d in train_split.iterdir() if d.is_dir()]) if train_split.exists() else []
    if len(class_dirs) < 2:
        return None

    class_names = [d.name for d in class_dirs]
    train_paths, train_labels = _list_split(classify_root / "train", class_names)
    val_paths, val_labels = _list_split(classify_root / "val", class_names)
    test_paths, test_labels = _list_split(classify_root / "test", class_names)

    return class_names, (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def _fallback_stratified_split(config: QuantumCNNConfig):
    """Recreate splits from raw data using existing repository helper functions."""
    from pathlib import Path
    dataset_root = Path(config.dataset_root) if isinstance(config.dataset_root, str) else config.dataset_root
    
    class_names = ["cracked", "not_cracked"]
    all_paths: List[Path] = []
    all_labels: List[int] = []

    for label, class_name in enumerate(class_names):
        class_paths = get_image_paths(dataset_root, class_name)
        all_paths.extend(class_paths)
        all_labels.extend([label] * len(class_paths))

    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = stratified_split(
        all_paths,
        all_labels,
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
        config.seed,
    )
    return class_names, (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def create_dataloaders(config: QuantumCNNConfig):
    """Create train/val/test dataloaders using shared split strategy."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    loaded = _load_from_existing_split(config)
    if loaded is None:
        class_names, train, val, test = _fallback_stratified_split(config)
    else:
        class_names, train, val, test = loaded

    train_paths, train_labels = _cap_split(train[0], train[1], config.max_train_samples, config.seed)
    val_paths, val_labels = _cap_split(val[0], val[1], config.max_val_samples, config.seed)
    test_paths, test_labels = _cap_split(test[0], test[1], config.max_test_samples, config.seed)

    transform = QuantumTransform(config)
    train_ds = PathLabelDataset(train_paths, train_labels, transform=transform)
    val_ds = PathLabelDataset(val_paths, val_labels, transform=transform)
    test_ds = PathLabelDataset(test_paths, test_labels, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, class_names
