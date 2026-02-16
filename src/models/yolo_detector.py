"""
YOLO-based crack detection and classification module.
Integrates Ultralytics YOLO for advanced object detection and segmentation.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import torch
import cv2
import numpy as np
from ultralytics import YOLO


class YOLOCrackDetector:
    """
    YOLO-based detector for concrete crack detection and classification.
    Supports YOLOv8 and beyond for both detection and segmentation tasks.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = 'yolov8n.pt',
        device: Optional[str] = None
    ):
        """
        Initialize YOLO crack detector.
        
        Args:
            model_path: Path to custom trained YOLO model
            model_type: Pre-trained YOLO model type (e.g., 'yolov8n.pt', 'yolov8s.pt')
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load YOLO model
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
            print(f"Loaded custom YOLO model from {model_path}")
        else:
            self.model = YOLO(model_type)
            print(f"Loaded pre-trained YOLO model: {model_type}")
        
        # Move model to device
        self.model.to(self.device)
    
    def detect_cracks(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        save_results: bool = False,
        output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect cracks in an image using YOLO.
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            save_results: Whether to save annotated results
            output_dir: Directory to save results
            
        Returns:
            List of detection dictionaries containing bbox, confidence, class
        """
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            save=save_results,
            project=output_dir,
            device=self.device
        )
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(box.conf[0]),
                    'class_id': int(box.cls[0]),
                    'class_name': result.names[int(box.cls[0])]
                }
                detections.append(detection)
        
        return detections
    
    def segment_cracks(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        save_results: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Segment cracks in an image using YOLO segmentation model.
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold
            save_results: Whether to save results
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing masks and metadata
        """
        # Note: Requires a YOLO segmentation model (e.g., yolov8n-seg.pt)
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            save=save_results,
            project=output_dir,
            device=self.device
        )
        
        # Parse segmentation results
        segmentation_data = {
            'masks': [],
            'boxes': [],
            'confidences': []
        }
        
        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                segmentation_data['masks'] = masks
                segmentation_data['boxes'] = boxes.tolist()
                segmentation_data['confidences'] = confs.tolist()
        
        return segmentation_data
    
    def batch_inference(
        self,
        image_dir: str,
        conf_threshold: float = 0.25,
        output_dir: str = "outputs/yolo_results"
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of images.
        
        Args:
            image_dir: Directory containing images
            conf_threshold: Confidence threshold
            output_dir: Directory to save results
            
        Returns:
            List of results for each image
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = list(image_dir.glob('*.jpg')) + \
                     list(image_dir.glob('*.png')) + \
                     list(image_dir.glob('*.jpeg'))
        
        batch_results = []
        
        for img_path in image_files:
            detections = self.detect_cracks(
                str(img_path),
                conf_threshold=conf_threshold,
                save_results=True,
                output_dir=str(output_dir)
            )
            
            batch_results.append({
                'image': img_path.name,
                'detections': detections,
                'num_cracks': len(detections)
            })
        
        return batch_results
    
    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        name: str = "crack_detection"
    ):
        """
        Train YOLO model on custom crack dataset.
        
        Args:
            data_yaml: Path to data configuration YAML file
            epochs: Number of training epochs
            imgsz: Input image size
            batch: Batch size
            name: Name for this training run
        """
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name=name,
            device=self.device
        )
        
        return results
    
    def export_model(
        self,
        format: str = 'onnx',
        output_path: Optional[str] = None
    ):
        """
        Export YOLO model to different formats.
        
        Args:
            format: Export format ('onnx', 'torchscript', 'tflite', etc.)
            output_path: Path to save exported model
        """
        self.model.export(format=format)
        print(f"Model exported to {format} format")
