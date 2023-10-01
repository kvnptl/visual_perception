
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torchmetrics
from torch.utils.tensorboard import SummaryWriter

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: tuple, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
  """Trains a PyTorch model for a single epoch.

  Turns a target PyTorch model to training mode and then
  runs through all of the required training steps (forward
  pass, loss calculation, optimizer step).

  Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: Tuple of two loss functions, one for classification and one for regression.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:
    
    (0.1112, 0.8743)
  """
  # Put model in train mode
  model.train()
  
  # Setup train loss and train accuracy values
  train_cls_loss, train_bbox_loss = 0.0, 0.0
  train_cls_acc, train_bbox_acc = 0.0, 0.0
  train_loss, train_acc = 0.0, 0.0

  # Weightagts for each loss function
  W1, W2 = 1.0, 1.0
  
  # Torchmetrics accuracy
  metric_cls = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
  metric_bbox = torchmetrics.detection.MeanAveragePrecision().to(device) # iou_thresholds are the stepped range [0.5,...,0.95] with step 0.05. Else provide a list of floats.
  
  # Loop through data loader data batches
  for batch, (X, y_cls, y_bbox) in enumerate(dataloader):
      # Send data to target device
      X, y_cls, y_bbox = X.to(device), y_cls.to(device), y_bbox.to(device)
      
      # 1. Forward pass
      y_pred = model(X)

      # 2. Calculate and accumulate loss
      classLoss = loss_fn[0](y_pred[0], y_cls)
      bboxLoss = loss_fn[1](y_pred[1], y_bbox)
      total_loss = W1 * classLoss + W2 * bboxLoss
      train_cls_loss += classLoss.item()
      train_bbox_loss += bboxLoss.item()
      train_loss += total_loss.item()
       
      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      total_loss.backward()

      # 5. Optimizer step
      optimizer.step()

      # Calculate and accumulate accuracy metric across all batches
      y_pred_class = torch.argmax(torch.softmax(y_pred[0], dim=1), dim=1)
      train_cls_acc += (y_pred_class == y_cls).sum().item()/len(y_pred[0]) # OPTION 1

      # Using torchmetric
      metric_cls.update(y_pred_class, y_cls) # OPTION 2

      # Prepare bbox predictions for torchmetric
      preds = [{'boxes': y_pred[1], 'scores': torch.tensor([1.0]*len(y_pred[1])), 'labels': y_pred_class}]
      targets = [{'boxes': y_bbox, 'labels': y_cls}]
      metric_bbox.update(preds, targets)

  # Adjust metrics to get average loss and accuracy per batch
  train_cls_loss = train_cls_loss / len(dataloader)
  train_cls_acc = train_cls_acc / len(dataloader) # OPTION 1

  # Cls acc using torchmetrics
  train_cls_acc_torchmetrics = metric_cls.compute() # OPTION 2
  metric_cls.reset()
  
  train_bbox_loss = train_bbox_loss / len(dataloader) 
  
  # Bbox acc using torchmetric
  train_bbox_acc_torchmetrics = metric_bbox.compute()["map"]
  metric_bbox.reset()

  # Total loss
  train_loss = train_loss / len(dataloader)

  # Total accuracy
  train_acc = ((train_cls_acc_torchmetrics + train_bbox_acc_torchmetrics) / 2).item() # item() works when a tensor contains a single value, otherwise use .cpu().numpy()

  return train_loss, train_acc, train_cls_loss, train_cls_acc_torchmetrics, train_bbox_loss, train_bbox_acc_torchmetrics

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: tuple,
              device: torch.device) -> Tuple[float, float]:
  """Tests a PyTorch model for a single epoch.

  Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a testing dataset.

  Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: Tuple of two loss functions, one for classification and one for regression.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:
    
    (0.0223, 0.8985)
  """
  # Put model in eval mode
  model.eval() 
  
  # Setup test loss and test accuracy values
  test_cls_loss, test_bbox_loss = 0.0, 0.0
  test_cls_acc = 0.0
  test_loss, test_acc = 0, 0

  # Weightagts for each loss function
  W1, W2 = 1.0, 1.0

  # Torchmetrics accuracy
  metric_cls = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
  metric_bbox = torchmetrics.detection.MeanAveragePrecision().to(device) # iou_thresholds are the stepped range [0.5,...,0.95] with step 0.05. Else provide a list of floats.
  
  # Turn on inference context manager
  with torch.inference_mode():
      # Loop through DataLoader batches
      for batch, (X, y_cls, y_bbox) in enumerate(dataloader):
          # Send data to target device
          X, y_cls, y_bbox = X.to(device), y_cls.to(device), y_bbox.to(device)
  
          # 1. Forward pass
          y_pred_test = model(X)

          # 2. Calculate and accumulate loss
          classLoss = loss_fn[0](y_pred_test[0], y_cls)
          bboxLoss = loss_fn[1](y_pred_test[1], y_bbox)
          total_loss = W1 * classLoss + W2 * bboxLoss
          
          test_cls_loss += classLoss.item()
          test_bbox_loss += bboxLoss.item()
          test_loss += total_loss.item()
          
          # Calculate and accumulate accuracy
          y_pred_class = torch.argmax(torch.softmax(y_pred_test[0], dim=1), dim=1)
          test_cls_acc += (y_pred_class == y_cls).sum().item()/len(y_pred_test[0])

          # Using torchmetric
          metric_cls.update(y_pred_class, y_cls) # OPTION 2

          # Prepare bbox predictions for torchmetric
          preds = [{'boxes': y_pred_test[1], 'scores': torch.tensor([1.0]*len(y_pred_test[1])), 'labels': y_pred_class}]
          targets = [{'boxes': y_bbox, 'labels': y_cls}]
          metric_bbox.update(preds, targets)
          
  # Adjust metrics to get average loss and accuracy per batch 
  test_cls_acc = test_cls_acc / len(dataloader)
  test_cls_loss = test_cls_loss / len(dataloader)

  # Using torchmetric
  test_cls_acc_torchmetrics = metric_cls.compute().item() # OPTION 2
  metric_cls.reset()

  test_bbox_acc_torchmetrics = metric_bbox.compute()["map"].item()  # item() works when a tensor contains a single value (scalar), otherwise use .cpu().numpy()
  test_bbox_loss = test_bbox_loss / len(dataloader)

  # Total loss
  test_loss = test_loss / len(dataloader)

  # Total accuracy
  test_acc = (test_cls_acc_torchmetrics + test_bbox_acc_torchmetrics) / 2

  return test_loss, test_acc, test_cls_loss, test_cls_acc_torchmetrics, test_bbox_loss, test_bbox_acc_torchmetrics

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: tuple, 
          epochs: int,
          device: torch.device,
          writer: SummaryWriter) -> Dict[str, List[float]]:
  """Trains and tests a PyTorch model.

  Passes a target PyTorch models through train_step() and test_step()
  functions for a number of epochs, training and testing the model
  in the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: Tuple of two loss functions, one for classification and one for regression.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]} 
    For example if training for epochs=2: 
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]} 
  """
  # Create empty results dictionary
  results = {"train_loss": [],
      "train_cls_loss": [],
      "train_bbox_loss": [],
      "train_acc": [],
      "valid_loss": [],
      "valid_cls_loss": [],
      "valid_bbox_loss": [],
      "valid_acc": [],
  }
  
  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      train_loss, train_acc, train_cls_loss, train_cls_acc, train_bbox_loss, train_bbox_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
      test_loss, test_acc, test_cls_loss, test_cls_acc, test_bbox_loss, test_bbox_acc  = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)
      
      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_cls_loss: {train_cls_loss:.4f} | "
          f"train_bbox_loss: {train_bbox_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"valid_loss: {test_loss:.4f} | "
          f"valid_cls_loss: {test_cls_loss:.4f} | "
          f"valid_bbox_loss: {test_bbox_loss:.4f} | "
          f"valid_acc: {test_acc:.4f}"
      )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_cls_loss"].append(train_cls_loss)
      results["train_bbox_loss"].append(train_bbox_loss)
      results["train_acc"].append(train_acc)
      results["valid_loss"].append(test_loss)
      results["valid_cls_loss"].append(test_cls_loss)
      results["valid_bbox_loss"].append(test_bbox_loss)
      results["valid_acc"].append(test_acc)

      if writer:
          writer.add_scalars(main_tag="Loss",
                            tag_scalar_dict={"train_loss": train_loss,
                                            "valid_loss": test_loss},
                            global_step=epoch)
          writer.add_scalars(main_tag="Class Loss",
                            tag_scalar_dict={"train_cls_loss": train_cls_loss,
                                            "valid_cls_loss": test_cls_loss},
                            global_step=epoch)
          writer.add_scalars(main_tag="Box Loss",
                            tag_scalar_dict={"train_bbox_loss": train_bbox_loss,
                                            "valid_bbox_loss": test_bbox_loss},
                            global_step=epoch)
          writer.add_scalars(main_tag="Accuracy",
                            tag_scalar_dict={"train_acc": train_acc,
                                            "valid_acc": test_acc},
                            global_step=epoch)
          
          writer.close()
        
      else:
         pass

  # Return the filled results at the end of the epochs
  return results
