"""
Image preprocessing utilities for crack detection.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union
import torch
from torchvision import transforms


class ImagePreprocessor:
    """
    Utility class for preprocessing concrete crack images.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target image size (height, width)
        """
        self.target_size = target_size
    
    def resize_image(
        self,
        image: Union[np.ndarray, Image.Image],
        size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image (numpy array or PIL Image)
            size: Target size, defaults to self.target_size
            
        Returns:
            Resized image as numpy array
        """
        size = size or self.target_size
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def denoise_image(
        self,
        image: np.ndarray,
        method: str = 'bilateral'
    ) -> np.ndarray:
        """
        Apply denoising to image.
        
        Args:
            image: Input image
            method: Denoising method ('bilateral', 'gaussian', 'median')
            
        Returns:
            Denoised image
        """
        if method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'median':
            return cv2.medianBlur(image, 5)
        else:
            raise ValueError(f"Unknown denoising method: {method}")
    
    def enhance_contrast(
        self,
        image: np.ndarray,
        method: str = 'clahe'
    ) -> np.ndarray:
        """
        Enhance image contrast.
        
        Args:
            image: Input image
            method: Enhancement method ('clahe', 'histogram_eq')
            
        Returns:
            Contrast-enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space for better results
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            if method == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
            elif method == 'histogram_eq':
                l = cv2.equalizeHist(l)
            
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            if method == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                return clahe.apply(image)
            elif method == 'histogram_eq':
                return cv2.equalizeHist(image)
    
    def apply_morphology(
        self,
        mask: np.ndarray,
        operation: str = 'close',
        kernel_size: int = 5
    ) -> np.ndarray:
        """
        Apply morphological operations to binary mask.
        
        Args:
            mask: Binary mask
            operation: Operation type ('open', 'close', 'dilate', 'erode')
            kernel_size: Kernel size for operation
            
        Returns:
            Processed mask
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )
        
        if operation == 'open':
            return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        elif operation == 'close':
            return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        elif operation == 'dilate':
            return cv2.dilate(mask, kernel, iterations=1)
        elif operation == 'erode':
            return cv2.erode(mask, kernel, iterations=1)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def edge_detection(
        self,
        image: np.ndarray,
        method: str = 'canny',
        **kwargs
    ) -> np.ndarray:
        """
        Apply edge detection to image.
        
        Args:
            image: Input image
            method: Edge detection method ('canny', 'sobel')
            **kwargs: Additional parameters for edge detection
            
        Returns:
            Edge map
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if method == 'canny':
            low_threshold = kwargs.get('low_threshold', 50)
            high_threshold = kwargs.get('high_threshold', 150)
            return cv2.Canny(gray, low_threshold, high_threshold)
        
        elif method == 'sobel':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            return np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
        
        else:
            raise ValueError(f"Unknown edge detection method: {method}")


def get_augmentation_transforms(image_size: Tuple[int, int] = (640, 640)):
    """
    Get augmentation transforms for training.
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transforms with augmentation
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy image.
    
    Args:
        tensor: Input tensor (C, H, W) or (B, C, H, W)
        
    Returns:
        Numpy image array (H, W, C) or (B, H, W, C)
    """
    if len(tensor.shape) == 4:
        # Batch of images
        images = tensor.cpu().detach().numpy()
        images = np.transpose(images, (0, 2, 3, 1))
    else:
        # Single image
        images = tensor.cpu().detach().numpy()
        images = np.transpose(images, (1, 2, 0))
    
    # Denormalize if needed
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = images * std + mean
    
    # Clip to valid range
    images = np.clip(images, 0, 1)
    
    return (images * 255).astype(np.uint8)
