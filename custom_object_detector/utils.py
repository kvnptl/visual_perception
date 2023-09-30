import numpy as np
import os
import torch
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision import transforms as T
from pathlib import Path

def xywh_to_xyxy(xywh):
    """
    Convert XYWH format (x,y center point and width, height) to XYXY format (x,y top left and x,y bottom right).
    :param xywh: [X, Y, W, H]
    :return: [X1, Y1, X2, Y2]
    """
    if np.array(xywh).ndim > 1 or len(xywh) > 4:
        raise ValueError('xywh format: [x1, y1, width, height]')
    xywh = [float(value) for value in xywh]
    x1 = xywh[0] - xywh[2] / 2
    y1 = xywh[1] - xywh[3] / 2
    x2 = xywh[0] + xywh[2] / 2
    y2 = xywh[1] + xywh[3] / 2
    return x1, y1, x2, y2

def readFile(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return lines

def set_seed(seed=8):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    cls_loss = results["train_cls_loss"]
    test_cls_loss = results["test_cls_loss"]

    bbox_loss = results["train_bbox_loss"]
    test_bbox_loss = results["test_bbox_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 5))

    # Plot loss
    plt.subplot(1, 4, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid()

    # Plot accuracy
    plt.subplot(1, 4, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid()

    # Plot cls loss
    plt.subplot(1, 4, 3)
    plt.plot(epochs, cls_loss, label="train_cls_loss")
    plt.plot(epochs, test_cls_loss, label="test_cls_loss")
    plt.title("Cls Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid()

    # Plot bbox loss
    plt.subplot(1, 4, 4)
    plt.plot(epochs, bbox_loss, label="train_bbox_loss")
    plt.plot(epochs, test_bbox_loss, label="test_bbox_loss")
    plt.title("Bbox Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid()

def create_confusion_matrix(model, test_loader, class_names, device):

  # Track predictions and true labels
  y_true = []
  y_pred = []
  
  # Make predictions on test set
  with torch.no_grad():
    for (X, y_cls, y_bbox) in test_loader:
        X, y_cls, y_bbox = X.to(device), y_cls.to(device), y_bbox.to(device)
        output = model(X)
        _, preds = torch.max(output[0], dim=1)
        y_true.extend(y_cls.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
      
  # Calculate confusion matrix
  conf_mat = confusion_matrix(y_true, y_pred)

  # Plot confusion matrix
  plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  plt.xticks(np.arange(len(class_names)), class_names, rotation='vertical')
  plt.yticks(np.arange(len(class_names)), class_names)
  plt.savefig('conf_mat.png')

from typing import List, Tuple
from PIL import Image
from torchvision import transforms

def pred_and_plot_img(model,
                      img_path,
                      class_names,
                      img_size,
                      transform,
                      device):

    img = Image.open(img_path)

    if transform is not None:
        img_transform = transform
    else:
        img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    model.to(device)

    model.eval()
    
    with torch.inference_mode():
        img_tensor = img_transform(img).unsqueeze(0)
        img_tensor = img_tensor.to(device)
        prediction = model(img_tensor)

    pred_probs = torch.softmax(prediction, dim=1)
    pred_class = torch.argmax(pred_probs, dim=1)

    plt.figure()
    plt.imshow(img)
    plt.title(f"Prediction: {class_names[pred_class]} | Prob: {pred_probs.max():.3f}")

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
  
  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)
  
  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)