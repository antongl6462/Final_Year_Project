"""
Mask generation pipeline for crack segmentation.
Creates binary masks highlighting concrete cracks.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union, Tuple
from PIL import Image

from ..models.model_loader import UNet
from ..utils.preprocessing import ImagePreprocessor


class MaskGenerator:
    """
    Generate segmentation masks for concrete cracks.
    Supports both traditional CV methods and deep learning models.
    """
    
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        device: Optional[str] = None
    ):
        """
        Initialize mask generator.
        
        Args:
            model: Optional pre-trained segmentation model
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = ImagePreprocessor()
        
        if self.model:
            self.model.to(self.device)
            self.model.eval()
    
    def generate_mask_threshold(
        self,
        image: np.ndarray,
        method: str = 'otsu'
    ) -> np.ndarray:
        """
        Generate mask using thresholding methods.
        
        Args:
            image: Input image (BGR format)
            method: Thresholding method ('otsu', 'adaptive', 'binary')
            
        Returns:
            Binary mask
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply preprocessing
        gray = self.preprocessor.denoise_image(gray, method='bilateral')
        
        if method == 'otsu':
            _, mask = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        elif method == 'adaptive':
            mask = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
        elif method == 'binary':
            _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        else:
            raise ValueError(f"Unknown thresholding method: {method}")
        
        # Post-process mask
        mask = self.preprocessor.apply_morphology(mask, operation='close', kernel_size=5)
        mask = self.preprocessor.apply_morphology(mask, operation='open', kernel_size=3)
        
        return mask
    
    def generate_mask_edges(
        self,
        image: np.ndarray,
        low_threshold: int = 50,
        high_threshold: int = 150
    ) -> np.ndarray:
        """
        Generate mask using edge detection.
        
        Args:
            image: Input image (BGR format)
            low_threshold: Lower threshold for Canny edge detection
            high_threshold: Upper threshold for Canny edge detection
            
        Returns:
            Binary edge mask
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Enhance contrast
        gray = self.preprocessor.enhance_contrast(gray, method='clahe')
        
        # Detect edges
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # Dilate edges to create thicker lines
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(edges, kernel, iterations=1)
        
        return mask
    
    def generate_mask_model(
        self,
        image: Union[str, np.ndarray, Image.Image],
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Generate mask using deep learning model.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            threshold: Threshold for binary mask (0-1)
            
        Returns:
            Binary mask
        """
        if self.model is None:
            raise ValueError("No model loaded. Cannot generate mask using model.")
        
        # Load and preprocess image
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        original_size = image.shape[:2]
        
        # Convert to tensor
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(image_tensor)
            
            # Apply sigmoid for binary segmentation
            if output.shape[1] == 1:
                pred_mask = torch.sigmoid(output)
            else:
                pred_mask = torch.softmax(output, dim=1)[:, 1:2, :, :]
            
            pred_mask = pred_mask.squeeze().cpu().numpy()
        
        # Threshold to binary mask
        binary_mask = (pred_mask > threshold).astype(np.uint8) * 255
        
        # Resize back to original size
        binary_mask = cv2.resize(binary_mask, (original_size[1], original_size[0]))
        
        return binary_mask
    
    def generate_mask_hybrid(
        self,
        image: np.ndarray,
        use_model: bool = True
    ) -> np.ndarray:
        """
        Generate mask using hybrid approach (combining multiple methods).
        
        Args:
            image: Input image (BGR format)
            use_model: Whether to use deep learning model
            
        Returns:
            Combined binary mask
        """
        # Generate masks using different methods
        mask_threshold = self.generate_mask_threshold(image, method='otsu')
        mask_edges = self.generate_mask_edges(image)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(mask_threshold, mask_edges)
        
        # If model is available, use it for refinement
        if use_model and self.model is not None:
            mask_model = self.generate_mask_model(image)
            # Weighted combination
            combined_mask = cv2.addWeighted(
                combined_mask, 0.5, mask_model, 0.5, 0
            ).astype(np.uint8)
        
        # Final post-processing
        combined_mask = self.preprocessor.apply_morphology(
            combined_mask, operation='close', kernel_size=5
        )
        
        return combined_mask
    
    def batch_generate(
        self,
        image_dir: str,
        output_dir: str,
        method: str = 'threshold'
    ):
        """
        Generate masks for a batch of images.
        
        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save generated masks
            method: Mask generation method ('threshold', 'edges', 'model', 'hybrid')
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = list(image_dir.glob('*.jpg')) + \
                     list(image_dir.glob('*.png')) + \
                     list(image_dir.glob('*.jpeg'))
        
        for img_path in image_files:
            # Load image
            image = cv2.imread(str(img_path))
            
            # Generate mask based on method
            if method == 'threshold':
                mask = self.generate_mask_threshold(image)
            elif method == 'edges':
                mask = self.generate_mask_edges(image)
            elif method == 'model':
                mask = self.generate_mask_model(image)
            elif method == 'hybrid':
                mask = self.generate_mask_hybrid(image)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Save mask
            output_path = output_dir / f"{img_path.stem}_mask{img_path.suffix}"
            cv2.imwrite(str(output_path), mask)
            print(f"Generated mask: {output_path}")
