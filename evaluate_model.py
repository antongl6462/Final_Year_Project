#!/usr/bin/env python3
"""
Standalone evaluation script for concrete crack detection model.
Evaluates the trained model and generates performance metrics.
"""

import subprocess
import sys

# Fix numpy compatibility first
try:
    import numpy
    if numpy.__version__.startswith('2.'):
        print("Downgrading numpy for PyTorch compatibility...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy<2"])
except:
    pass

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score
)

# Add project to path
sys.path.insert(0, '/Users/anton/Documents/GitHub/Final_Year_Project')

from src.models import ClassificationModel

def load_and_preprocess_image(img_path, size=(227, 227)):
    """Load and preprocess image for model inference"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize(size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
    return img_tensor

def evaluate_model(model, test_files, test_labels, device, sample_size=None):
    """Evaluate model on test set"""
    
    if sample_size is None:
        sample_size = len(test_files)
    else:
        sample_size = min(sample_size, len(test_files))
    
    y_true, y_pred, y_prob = [], [], []
    
    model.eval()
    with torch.no_grad():
        for idx in range(sample_size):
            if (idx + 1) % 500 == 0:
                print(f"  Processed: {idx + 1}/{sample_size} images")
            
            try:
                img_tensor = load_and_preprocess_image(test_files[idx])
                img_tensor = img_tensor.unsqueeze(0).to(device)
                
                logits = model(img_tensor)
                prob = torch.sigmoid(logits).cpu().item()
                
                y_true.append(test_labels[idx])
                y_pred.append(1 if prob > 0.5 else 0)
                y_prob.append(prob)
                
            except Exception as e:
                print(f"  Error processing image {idx} ({test_files[idx]}): {e}")
                continue
    
    return y_true, y_pred, y_prob

def main():
    """Main evaluation function"""
    
    print("\n" + "="*80)
    print("CONCRETE CRACK DETECTION - MODEL EVALUATION")
    print("="*80 + "\n")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    DATA_ROOT = '/Users/anton/Documents/GitHub/Final_Year_Project/concrete-crack-images-for-classification'
    
    test_files = []
    test_labels = []
    for label_idx, class_dir in enumerate(['Negative', 'Positive']):
        class_path = os.path.join(DATA_ROOT, class_dir)
        for img_file in os.listdir(class_path):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                test_files.append(os.path.join(class_path, img_file))
                test_labels.append(label_idx)
    
    print(f"Total test images: {len(test_files)}")
    print(f"  Negative (Safe):    {sum(1 for l in test_labels if l == 0)}")
    print(f"  Positive (Cracked): {sum(1 for l in test_labels if l == 1)}\n")
    
    # Load model
    print("Loading model...")
    model = ClassificationModel()
    # Try to load checkpoint
    checkpoint_path = '/Users/anton/Documents/GitHub/Final_Year_Project/crack_detection_model.pth'
    if os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"✓ Model loaded from checkpoint: {checkpoint_path}\n")
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
            print("   Using untrained model\n")
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("   Using untrained model\n")
    
    model = model.to(device)
    
    # Evaluate
    print("Evaluating model on test set...")
    y_true, y_pred, y_prob = evaluate_model(model, test_files, test_labels, device)
    
    print(f"\n✓ Evaluation complete!")
    print(f"  Successfully evaluated: {len(y_true)} images\n")
    
    if len(y_true) == 0:
        print("❌ No images were successfully evaluated!")
        return
    
    # Calculate metrics
    print("="*80)
    print("PERFORMANCE METRICS")
    print("="*80 + "\n")
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    print(f"📊 Classification Metrics:")
    print(f"  • Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  • Precision:   {precision:.4f}")
    print(f"  • Recall:      {recall:.4f}")
    print(f"  • F1 Score:    {f1:.4f}")
    print(f"  • ROC-AUC:     {roc_auc:.4f}\n")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    
    print(f"📈 Confusion Matrix:")
    print(f"                  Predicted Negative    Predicted Positive")
    print(f"Actual Negative:       {tn:6d}               {fp:6d}")
    print(f"Actual Positive:       {fn:6d}               {tp:6d}\n")
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"📋 Additional Metrics:")
    print(f"  • Sensitivity (TPR): {sensitivity:.4f}")
    print(f"  • Specificity (TNR): {specificity:.4f}")
    print(f"  • False Positive Rate: {fp/(fp+tn) if (fp+tn) > 0 else 0:.4f}")
    print(f"  • False Negative Rate: {fn/(fn+tp) if (fn+tp) > 0 else 0:.4f}\n")
    
    print("="*80)
    print("CLASSIFICATION REPORT:")
    print("="*80)
    target_names = ['Negative (Safe)', 'Positive (Cracked)']
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    
    # Visualizations
    print("\n📊 Generating performance visualizations...\n")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=target_names, yticklabels=target_names,
                cbar_kws={'label': 'Count'})
    axes[0, 0].set_title('Confusion Matrix (Absolute)', fontsize=13, fontweight='bold')
    axes[0, 0].set_ylabel('True Label', fontweight='bold')
    axes[0, 0].set_xlabel('Predicted Label', fontweight='bold')
    
    # 2. Normalized Confusion Matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn', ax=axes[0, 1],
                xticklabels=target_names, yticklabels=target_names,
                vmin=0, vmax=1, cbar_kws={'label': 'Percentage'})
    axes[0, 1].set_title('Confusion Matrix (Normalized)', fontsize=13, fontweight='bold')
    axes[0, 1].set_ylabel('True Label', fontweight='bold')
    axes[0, 1].set_xlabel('Predicted Label', fontweight='bold')
    
    # 3. Metrics Bar Chart
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    metrics_values = [accuracy, precision, recall, f1, roc_auc]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = axes[1, 0].bar(metrics_names, metrics_values, color=colors, alpha=0.75,
                          edgecolor='black', linewidth=1.5)
    axes[1, 0].set_ylim([0, 1.15])
    axes[1, 0].set_ylabel('Score', fontweight='bold')
    axes[1, 0].set_title('Performance Metrics Summary', fontsize=13, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1, 0].set_axisbelow(True)
    
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)
    
    axes[1, 1].plot(fpr, tpr, color='darkorange', lw=3,
                    label=f'ROC Curve (AUC = {roc_auc_val:.3f})')
    axes[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                    alpha=0.7, label='Random Classifier (AUC = 0.500)')
    axes[1, 1].fill_between(fpr, tpr, alpha=0.2, color='darkorange')
    axes[1, 1].set_xlim([-0.02, 1.02])
    axes[1, 1].set_ylim([-0.02, 1.02])
    axes[1, 1].set_xlabel('False Positive Rate', fontweight='bold')
    axes[1, 1].set_ylabel('True Positive Rate', fontweight='bold')
    axes[1, 1].set_title('ROC Curve Analysis', fontsize=13, fontweight='bold')
    axes[1, 1].legend(loc="lower right", fontsize=10)
    axes[1, 1].grid(alpha=0.3, linestyle='--')
    axes[1, 1].set_axisbelow(True)
    
    plt.suptitle('Concrete Crack Detection - Model Performance Benchmark',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('/Users/anton/Documents/GitHub/Final_Year_Project/performance_benchmark.png', dpi=150, bbox_inches='tight')
    print("✓ Performance visualization saved to: performance_benchmark.png\n")
    plt.show()
    
    # Summary
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total Samples Evaluated: {len(y_true)}")
    print(f"\nClass Distribution:")
    print(f"  Negative (Safe):    {(np.array(y_true) == 0).sum()} samples")
    print(f"  Positive (Cracked): {(np.array(y_true) == 1).sum()} samples")
    print(f"\nCorrectly Classified:")
    print(f"  True Negatives:  {tn} ({tn/(tn+fp)*100:.1f}% of negatives)")
    print(f"  True Positives:  {tp} ({tp/(tp+fn)*100:.1f}% of positives)")
    print(f"\nMisclassified:")
    print(f"  False Positives: {fp} (False Alarms)")
    print(f"  False Negatives: {fn} (Missed Detections)")
    print("\n" + "="*80)
    print("✓ EVALUATION COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
