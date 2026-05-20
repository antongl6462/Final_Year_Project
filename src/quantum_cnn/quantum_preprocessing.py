"""Quantum-specific lightweight preprocessing utilities.

These transforms are optional and only used by the hybrid QCNN branch. The
classical YOLO pipeline remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class QuantumPreprocessConfig:
    """Controls optional crack-enhancing preprocessing."""

    use_grayscale: bool = True
    use_clahe: bool = True
    use_blackhat: bool = False


def preprocess_for_quantum(image: Image.Image, cfg: QuantumPreprocessConfig) -> Image.Image:
    """Apply lightweight preprocessing before tensor conversion.

    Steps are intentionally simple for simulation-time practicality.
    """
    array = np.array(image.convert("RGB"))
    if cfg.use_grayscale:
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

    if cfg.use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    if cfg.use_blackhat:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        gray = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    return Image.fromarray(gray, mode="L")
