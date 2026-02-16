"""
Visualization utilities for crack detection results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


def visualize_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    show: bool = True
) -> np.ndarray:
    """
    Visualize bounding box detections on image.
    
    Args:
        image: Input image (BGR format)
        detections: List of detection dictionaries with 'bbox', 'confidence', 'class_name'
        save_path: Optional path to save visualization
        show: Whether to display the image
        
    Returns:
        Annotated image
    """
    annotated = image.copy()
    
    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        confidence = det['confidence']
        class_name = det.get('class_name', 'crack')
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(
            annotated,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            (0, 255, 0),
            -1
        )
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2
        )
    
    if save_path:
        cv2.imwrite(save_path, annotated)
    
    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f'Detected Cracks: {len(detections)}')
        plt.tight_layout()
        plt.show()
    
    return annotated


def visualize_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (0, 255, 0),
    save_path: Optional[str] = None,
    show: bool = True
) -> np.ndarray:
    """
    Overlay segmentation mask on image.
    
    Args:
        image: Input image (BGR format)
        mask: Binary mask (0-1 or 0-255)
        alpha: Transparency of overlay
        color: Color for mask overlay (BGR)
        save_path: Optional path to save visualization
        show: Whether to display the image
        
    Returns:
        Image with mask overlay
    """
    # Ensure mask is binary
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 127] = color
    
    # Blend with original image
    overlaid = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    
    if save_path:
        cv2.imwrite(save_path, overlaid)
    
    if show:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Crack Mask')
        axes[1].axis('off')
        
        axes[2].imshow(cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return overlaid


def create_comparison_grid(
    images: List[np.ndarray],
    titles: List[str],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Create a grid comparison of multiple images.
    
    Args:
        images: List of images to display
        titles: List of titles for each image
        save_path: Optional path to save the grid
        show: Whether to display the grid
    """
    n_images = len(images)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    
    if n_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 3 and img.shape[2] == 3:
            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[idx].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axes[idx].set_title(title)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training history metrics.
    
    Args:
        history: Dictionary with 'loss', 'val_loss', 'accuracy', 'val_accuracy'
        save_path: Optional path to save plot
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    if 'loss' in history:
        axes[0].plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    if 'accuracy' in history:
        axes[1].plot(history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history:
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
