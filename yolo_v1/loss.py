import torch
import torch.nn as nn
from utils import intersection_over_union

class YOLOLoss(nn.Module):
    def __init__(self, split_size=7, num_boxes=2, num_classes=20):
        super().__init__()
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        self.mse = nn.MSELoss(reduction="sum") # sum of all elements
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # We have to reshape as YOLO output is (1, 1470), so convert it to (1, 7, 7, 30)
        predictions = predictions.reshape(-1, 
                                         self.split_size, 
                                         self.split_size, 
                                         self.num_classes + self.num_boxes * 5) # -1 means don't touch the first dimension, it could be anything (here batch size)

        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25]) # [...]: means take all elements, e.g. from (1, 7, 7, 30), take (1, 7, 7) as it is
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3).float() # Identity matrix

        # Bounding box regression
        box_predictions = exists_box * (
            best_box * predictions[..., 26:30] + (1 - best_box) * predictions[..., 21:25]
        )
        box_targets = exists_box * target[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * \
                                    torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6)) # sign is used to preserve the sign, as we are removing while calculating square root

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )


        # Object loss
        pred_box = (
            best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21]
        )

        # (N, S, S) -> (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # No object loss
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1), 
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1), 
        )

        # Class loss
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2), # end_dim=-2: flatten the last dim (N, S, S, 20) -> (N*S*S, 20)
            torch.flatten(exists_box * target[..., :20], end_dim=-2),
        )

        # Total loss
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss