import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

def threshold_predictions(preds, threshold=0.5):
    return (preds > threshold).astype(np.uint8)

def classification_metrics(y_true, y_pred, y_prob=None):
    metrics = {}
    metrics['f1'] = f1_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    return metrics

def dice_coef(pred, target, threshold=0.5, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    target = target.float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean().item()

def iou_coef(pred, target, threshold=0.5, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    target = target.float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()
