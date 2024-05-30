import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(
            inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss

        if self.reduction == 'sum':
            return F_loss.sum()
        elif self.reduction == 'mean':
            return F_loss.mean()
    
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        # Apply softmax to the predictions
        pred = F.softmax(pred, dim=1)
        
        # Convert target to one-hot encoding
        target = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        # Handle ignore index
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            pred = pred * mask
            target = target * mask
            
            # Zero out the predictions and targets where mask is False
            pred = pred * mask.float()
            target = target * mask.float()

        # Calculate intersection and union
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

        # Compute Dice coefficient
        dice = 2.0 * intersection / (union + self.smooth)
        
        # Return Dice loss
        return 1.0 - dice.mean()

def tversky_loss(pred, target, alpha=0.5, beta=0.5, smooth=1e-6):
    pred = torch.sigmoid(pred)
    tp = (pred * target).sum(dim=(2, 3))
    fp = ((1 - target) * pred).sum(dim=(2, 3))
    fn = (target * (1 - pred)).sum(dim=(2, 3))
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky.mean()


def boundary_loss(pred, target, b_weight=1.0, smooth=1e-6):
    pred = torch.sigmoid(pred)
    boundary_target = target - F.avg_pool2d(target, 3, stride=1, padding=1)
    boundary_pred = pred - F.avg_pool2d(pred, 3, stride=1, padding=1)
    b_loss = F.binary_cross_entropy(boundary_pred, boundary_target)
    return b_weight * b_loss
