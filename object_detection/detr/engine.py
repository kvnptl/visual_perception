
import torch
import torchmetrics
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

import numpy as np

import utils_detr

# Initialize the MeanAveragePrecision object
metric = torchmetrics.detection.mean_ap.MeanAveragePrecision(
    box_format='xyxy', 
    iou_type='bbox', 
    iou_thresholds=np.arange(0.5, 1.0, 0.05).tolist(),  # specify the IoU thresholds here
    rec_thresholds=None, 
    max_detection_thresholds=None, 
    class_metrics=False, 
    extended_summary=False, 
    average='macro', 
    backend='pycocotools'
)

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
  """Trains a PyTorch model for a single epoch.

  Turns a target PyTorch model to training mode and then
  runs through all of the required training steps (forward
  pass, loss calculation, optimizer step).

  Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:
    
    (0.1112, 0.8743)
  """
  # Put model in train mode
  model.train()
  loss_fn.train()
  
  # Setup train loss and train accuracy values
  train_loss, train_acc = 0.0, 0.0
  
  # Loop through data loader data batches
  for batch, (X, y) in enumerate(dataloader):
    # Send data to target device
    X = torch.stack([img.to(device) for img in X])
    targets = [{k: v.to(device) for k, v in t.items()} for t in y]

    # 1. Forward pass
    y_pred = model(X)

    # 2. Calculate  and accumulate loss
    loss_dict = loss_fn(y_pred, targets)
    weight_dict = loss_fn.weight_dict
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
    train_loss += losses.item() 

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    losses.backward()

    # 5. Optimizer step
    optimizer.step()

    # 6. Calculate mAP
    # Calculate mAP
    # Convert box format
    y_pred_boxes_coco = y_pred['pred_boxes'].detach()
    targets_boxes_coco = [target['boxes'].detach() for target in targets]

    # Apply softmax to the last dimension of pred_logits
    y_pred_scores = torch.nn.functional.softmax(y_pred['pred_logits'], dim=-1)
    y_pred_scores, y_pred_labels = y_pred_scores.max(dim=-1)

    # Create prediction and target dictionaries
    preds_coco = []
    targets_coco = []
    for i in range(len(X)):
        preds_coco.append({
            "boxes": y_pred_boxes_coco[i].to(device),
            "scores": y_pred_scores[i].to(device),
            "labels": y_pred_labels[i].to(device)
        })
        targets_coco.append({
            "boxes": targets_boxes_coco[i].to(device),
            "labels": targets[i]['labels'].detach().to(device)
        })

    map_value = metric(preds=preds_coco, target=targets_coco)
    train_acc += map_value["map"].item()

  # Adjust metrics to get average loss and accuracy per batch 
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

def val_step(model: torch.nn.Module, 
       dataloader: torch.utils.data.DataLoader, 
       loss_fn: torch.nn.Module,
       device: torch.device) -> Tuple[float, float]:
  # Put model in eval mode
  model.eval()
  loss_fn.eval()
  
  # Setup val loss and val accuracy values
  val_loss, val_acc = 0, 0
  
  # Turn on inference context manager
  with torch.inference_mode():
    # Loop through DataLoader batches
    for batch, (X, y) in enumerate(dataloader):
      # Send data to target device
      X = torch.stack([img.to(device) for img in X])
      targets = [{k: v.to(device) for k, v in t.items()} for t in y]
  
      # 1. Forward pass
      val_pred = model(X)

      # 2. Calculate and accumulate loss
      loss_dict = loss_fn(val_pred, targets)
      weight_dict = loss_fn.weight_dict
      losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
      val_loss += losses.item()

      # Calculate mAP
      # Convert box format
      val_pred_boxes_coco = val_pred['pred_boxes'].detach()
      targets_boxes_coco = [target['boxes'].detach() for target in targets]

      # Apply softmax to the last dimension of pred_logits
      val_pred_scores = torch.nn.functional.softmax(val_pred['pred_logits'], dim=-1)
      val_pred_scores, val_pred_labels = val_pred_scores.max(dim=-1)

      # Create prediction and target dictionaries
      preds_coco = []
      targets_coco = []
      for i in range(len(X)):
        preds_coco.append({
          "boxes": val_pred_boxes_coco[i].to(device),
          "scores": val_pred_scores[i].to(device),
          "labels": val_pred_labels[i].to(device)
        })
        targets_coco.append({
          "boxes": targets_boxes_coco[i].to(device),
          "labels": targets[i]['labels'].detach().to(device)
        })

      map_value = metric(preds=preds_coco, target=targets_coco)
      val_acc += map_value["map"].item()
      
  # Adjust metrics to get average loss and accuracy per batch 
  val_loss = val_loss / len(dataloader)
  val_acc = val_acc / len(dataloader)
  return val_loss, val_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List[float]]:

  # Create empty results dictionary
  results = {"train_loss": [],
      "train_acc": [],
      "val_loss": [],
      "val_acc": []
  }
  
  # Loop through training and validation steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
      val_loss, val_acc = val_step(model=model,
          dataloader=val_dataloader,
          loss_fn=loss_fn,
          device=device)
      
      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"val_loss: {val_loss:.4f} | "
          f"val_acc: {val_acc:.4f}"
      )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["val_loss"].append(val_loss)
      results["val_acc"].append(val_acc)

  # Return the filled results at the end of the epochs
  return results
