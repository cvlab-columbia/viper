import torch
import torch.nn.functional as F

def bce_loss(pred, mask, reduction='none'):
    bce = F.binary_cross_entropy(pred, mask, reduction=reduction)
    return bce

def weighted_bce_loss(pred, mask, reduction='none'):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    weight = weight.flatten()
    
    bce = weight * bce_loss(pred, mask, reduction='none').flatten()
    
    if reduction == 'mean':
        bce = bce.mean()
    
    return bce

def iou_loss(pred, mask, reduction='none'):
    inter = pred * mask
    union = pred + mask
    iou = 1 - (inter + 1) / (union - inter + 1)

    if reduction == 'mean':
        iou = iou.mean()

    return iou

def bce_loss_with_logits(pred, mask, reduction='none'):
    return bce_loss(torch.sigmoid(pred), mask, reduction=reduction)

def weighted_bce_loss_with_logits(pred, mask, reduction='none'):
    return weighted_bce_loss(torch.sigmoid(pred), mask, reduction=reduction)

def iou_loss_with_logits(pred, mask, reduction='none'):
    return iou_loss(torch.sigmoid(pred), mask, reduction=reduction)