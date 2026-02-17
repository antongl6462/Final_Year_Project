import torch
import torch.nn as nn
import torch.nn.functional as F

def bce_dice_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    target = target.float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def focal_loss(pred, target, alpha=0.8, gamma=2):
    pred = torch.sigmoid(pred)
    target = target.float()
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce)
    loss = alpha * (1-pt) ** gamma * bce
    return loss.mean()

def get_loss(name, **kwargs):
    if name == 'bce':
        return nn.BCEWithLogitsLoss()
    elif name == 'dice':
        return dice_loss
    elif name == 'bce_dice':
        return bce_dice_loss
    elif name == 'focal':
        return focal_loss
    else:
        raise ValueError(f'Unknown loss: {name}')
