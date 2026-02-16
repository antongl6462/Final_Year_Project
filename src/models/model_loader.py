"""
Model loader and manager for importing pre-trained neural networks.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torchvision import models


class ModelLoader:
    """
    Utility class for loading and managing pre-trained models.
    Supports PyTorch models, ONNX, and custom architectures.
    """
    
    def __init__(self, model_dir: str = "models/downloaded"):
        """
        Initialize the model loader.
        
        Args:
            model_dir: Directory to store downloaded/loaded models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: Dict[str, nn.Module] = {}
    
    def load_pytorch_model(
        self,
        model_path: str,
        architecture: Optional[str] = None,
        num_classes: int = 2
    ) -> nn.Module:
        """
        Load a PyTorch model from a checkpoint file.
        
        Args:
            model_path: Path to the model checkpoint (.pth or .pt)
            architecture: Model architecture name (e.g., 'resnet50', 'unet')
            num_classes: Number of output classes
            
        Returns:
            Loaded PyTorch model
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # If architecture is specified, create model instance
        if architecture:
            model = self._create_model(architecture, num_classes)
            
            # Load weights
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            # Assume checkpoint is the full model
            model = checkpoint
        
        model.eval()
        return model
    
    def load_pretrained_backbone(
        self,
        architecture: str = 'resnet50',
        pretrained: bool = True,
        num_classes: int = 2
    ) -> nn.Module:
        """
        Load a pre-trained backbone model from torchvision.
        
        Args:
            architecture: Model architecture (resnet50, resnet101, etc.)
            pretrained: Whether to load ImageNet pre-trained weights
            num_classes: Number of output classes (modifies final layer)
            
        Returns:
            Pre-trained model
        """
        if architecture.lower() == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            # Modify final layer for binary/multi-class classification
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        elif architecture.lower() == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        elif architecture.lower() == 'vgg16':
            model = models.vgg16(pretrained=pretrained)
            model.classifier[6] = nn.Linear(4096, num_classes)
        
        elif architecture.lower() == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Cache the model
        self.loaded_models[architecture] = model
        
        return model
    
    def _create_model(self, architecture: str, num_classes: int) -> nn.Module:
        """
        Create a model instance based on architecture name.
        
        Args:
            architecture: Model architecture name
            num_classes: Number of output classes
            
        Returns:
            Model instance
        """
        return self.load_pretrained_backbone(
            architecture=architecture,
            pretrained=False,
            num_classes=num_classes
        )
    
    def save_model(
        self,
        model: nn.Module,
        save_path: str,
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """
        Save a model checkpoint.
        
        Args:
            model: PyTorch model to save
            save_path: Path to save the checkpoint
            additional_info: Additional information to save (epoch, optimizer state, etc.)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'state_dict': model.state_dict(),
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")


class UNet(nn.Module):
    """
    Simple U-Net architecture for image segmentation.
    Useful for generating crack masks from concrete images.
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        """
        Initialize U-Net model.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output channels (1 for binary mask)
        """
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a convolution block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net."""
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.out(dec1)
