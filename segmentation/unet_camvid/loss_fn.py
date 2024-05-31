import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Region-based loss functions
- Dice loss
- IoU loss
- Tversky loss

Pixel-based loss functions
- Cross entropy loss
- Focal loss
"""

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

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6, ignore_index=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
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

        # Calculate true positives, false positives and false negatives
        tp = (pred * target).sum(dim=(2, 3))
        fp = ((1 - target) * pred).sum(dim=(2, 3))
        fn = (target * (1 - pred)).sum(dim=(2, 3))

        # Compute Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Return Tversky loss
        return 1.0 - tversky.mean()

class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Apply softmax to the predictions
        pred = F.softmax(pred, dim=1)
        
        # Convert target to one-hot encoding
        target = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

        # Calculate intersection and union
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection

        # Compute IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Return IoU loss
        return 1.0 - iou.mean()

class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()

    def forward(self, pred, target):
        dice_loss = self.dice_loss(pred, target)
        focal_loss = self.focal_loss(pred, target)
        return self.alpha * dice_loss + self.beta * focal_loss
    
class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice_loss = DiceLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        dice_loss = self.dice_loss(pred, target)
        cross_entropy_loss = self.cross_entropy_loss(pred, target)
        return self.alpha * dice_loss + self.beta * cross_entropy_loss
    
class IoUCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.iou_loss = IoULoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        iou_loss = self.iou_loss(pred, target)
        cross_entropy_loss = self.cross_entropy_loss(pred, target)
        return self.alpha * iou_loss + self.beta * cross_entropy_loss
    
class IoUFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.iou_loss = IoULoss()
        self.focal_loss = FocalLoss()

    def forward(self, pred, target):
        iou_loss = self.iou_loss(pred, target)
        focal_loss = self.focal_loss(pred, target)
        return self.alpha * iou_loss + self.beta * focal_loss
    
class TverskyCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.tversky_loss = TverskyLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        tversky_loss = self.tversky_loss(pred, target)
        cross_entropy_loss = self.cross_entropy_loss(pred, target)
        return self.alpha * tversky_loss + self.beta * cross_entropy_loss
    
class TverskyFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.tversky_loss = TverskyLoss()
        self.focal_loss = FocalLoss()

    def forward(self, pred, target):
        tversky_loss = self.tversky_loss(pred, target)
        focal_loss = self.focal_loss(pred, target)
        return self.alpha * tversky_loss + self.beta * focal_loss
    
