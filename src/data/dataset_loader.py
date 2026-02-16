"""
Dataset management module for loading and preprocessing concrete crack datasets.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms


class CrackDataset(Dataset):
    """
    Custom dataset class for concrete crack images.
    Supports loading images and their corresponding masks for segmentation tasks.
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        image_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize the crack dataset.
        
        Args:
            image_dir: Directory containing input images
            mask_dir: Directory containing mask images (for segmentation)
            transform: Optional transformations to apply
            image_size: Target image size. Can be single int or (height, width) tuple
                       for compatibility with torchvision transforms
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform
        self.image_size = image_size
        
        # Get list of image files
        self.image_files = sorted([
            f for f in self.image_dir.glob("*")
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ])
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single item from the dataset.
        
        Returns:
            Dictionary containing 'image' and optionally 'mask'
        """
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load mask if available
        mask = None
        if self.mask_dir:
            mask_path = self.mask_dir / img_path.name
            if mask_path.exists():
                mask = Image.open(mask_path).convert('L')  # Grayscale
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            if mask:
                mask = self.transform(mask)
        else:
            # Default transform: resize and convert to tensor
            image = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ])(image)
            
            if mask:
                mask = transforms.Compose([
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                ])(mask)
        
        result = {
            'image': image,
            'image_path': str(img_path),
            'image_name': img_path.name
        }
        
        if mask is not None:
            result['mask'] = mask
        
        return result


class DatasetImporter:
    """
    Utility class for importing and organizing crack detection datasets.
    Supports various dataset formats commonly used in research.
    """
    
    @staticmethod
    def get_default_transforms(image_size: Tuple[int, int] = (640, 640)) -> transforms.Compose:
        """
        Get default image transformations for training.
        
        Args:
            image_size: Target image size. Can be single int or (height, width) tuple.
                       If tuple, uses first dimension for square resize.
            
        Returns:
            Composed transforms
        """
        # Use single size for square images
        if isinstance(image_size, tuple):
            resize_size = image_size[0]
        else:
            resize_size = image_size
            
        return transforms.Compose([
            transforms.Resize(resize_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @staticmethod
    def create_dataloader(
        dataset: Dataset,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 4
    ) -> DataLoader:
        """
        Create a DataLoader from a dataset.
        
        Args:
            dataset: PyTorch Dataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            
        Returns:
            DataLoader instance
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    @staticmethod
    def verify_dataset_structure(dataset_path: str) -> dict:
        """
        Verify the structure of a dataset directory.
        
        Args:
            dataset_path: Path to dataset root
            
        Returns:
            Dictionary with dataset statistics
        """
        dataset_path = Path(dataset_path)
        
        stats = {
            'exists': dataset_path.exists(),
            'num_images': 0,
            'num_masks': 0,
            'image_extensions': set(),
            'subdirectories': []
        }
        
        if not stats['exists']:
            return stats
        
        # Count files
        for item in dataset_path.rglob('*'):
            if item.is_file():
                ext = item.suffix.lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    stats['num_images'] += 1
                    stats['image_extensions'].add(ext)
            elif item.is_dir():
                stats['subdirectories'].append(item.name)
        
        stats['image_extensions'] = list(stats['image_extensions'])
        
        return stats
