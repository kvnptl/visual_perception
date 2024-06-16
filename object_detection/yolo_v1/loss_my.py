import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, grid_size=7, num_boxes=2, num_classes=20):
        super().__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_weight = [5.0, 0.5]
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, pred, target):

        loss = 0
        
        # Reshape pred to match target
        pred = pred.view(target.shape)

        # Extract data from pred and target for comparison
        target_class = target[..., :20] # 20 classes
        pred_class = pred[..., :20]

        target_objness = target[..., 20].unsqueeze(-1)
        pred1_objness = pred[..., 20].unsqueeze(-1)
        pred2_objness = pred[..., 25].unsqueeze(-1)

        target_bbox = target[..., 21:25]
        pred_bbox1 = pred[..., 21:25]
        pred_bbox2 = pred[..., 26:30]

        # Get the best bbox out of 2 boxes based on objectness score
        """
        NOTE: 
        torch.cat - concatenating tensors along an existing dimension, 
        torch.stack - stacking tensors along a new dimension
        """
        pred_boxes = torch.stack((pred1_objness, pred2_objness), dim=0)
        _, best_box_idx = torch.max(pred_boxes, dim=0)

        ##############
        # Class loss
        ##############
        cls_loss = self.mse_loss(target_objness*pred_class, target_class)

        ##############
        # Objectness loss
        ##############
        pred_best_objness = (1-best_box_idx) * pred1_objness + (best_box_idx) * pred2_objness
        obj_loss = self.mse_loss(target_objness*pred_best_objness, target_objness)

        ##############
        # No objectness loss
        ##############
        no_obj_loss1 = self.mse_loss((1-target_objness) * pred1_objness, (1-target_objness)*target_objness)
        no_obj_loss2 = self.mse_loss((1-target_objness) * pred2_objness, (1-target_objness)*target_objness)
        no_obj_loss = no_obj_loss1 + no_obj_loss2

        ##############
        # Bbox loss
        ##############
        pred_best_bbox = (1-best_box_idx) * pred_bbox1 + (best_box_idx) * pred_bbox2

        pred_bbox_xy = self.mse_loss(target_objness*pred_best_bbox[..., 0:2], target_bbox[..., 0:2])

        pred_bbox_wh_sign = torch.sign(pred_best_bbox[..., 2:4]) # Retain signs

        pred_bbox_wh = self.mse_loss(target_objness * pred_bbox_wh_sign * torch.sqrt(torch.abs(pred_best_bbox[..., 2:4])), torch.sqrt(torch.abs(target_bbox[..., 2:4])))
           
        bbox_loss = pred_bbox_xy + pred_bbox_wh
        
        # Final loss
        loss = self.lambda_weight[0] * bbox_loss + cls_loss + obj_loss + self.lambda_weight[1] * no_obj_loss

        return loss


def main():
    loss_fn = YOLOLoss()

    torch.manual_seed(0)

    # Dummy input
    pred = torch.randn([1, 7*7*(20+(2*5))]) # randn: Generates tensor from normal distribution, rand: from uniform distribution
    target = torch.randn([7, 7, (20+(2*5))]).unsqueeze(0)

    print(f"Shape pred: {pred.shape} and target shape: {target.shape}")

    # Loss fn
    loss = loss_fn(pred, target)

    print(f"Loss : {loss}")

if __name__ == "__main__":
    main()