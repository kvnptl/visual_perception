import numpy as np
import os
import torch
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision import transforms as T
from pathlib import Path

from typing import List, Tuple
from PIL import Image
import cv2
import config

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

def crawl_through_dir(dir_path):
    file_paths = []
    for root, directories, files in os.walk(dir_path):
        for filename in files: 
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    
    file_paths = sorted(file_paths)
    return file_paths

def set_seed(seed=8):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# Plot loss curves of a model
def plot_loss_curves(results, save_fig=False):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["valid_loss"]

    cls_loss = results["train_cls_loss"]
    test_cls_loss = results["valid_cls_loss"]

    bbox_loss = results["train_bbox_loss"]
    test_bbox_loss = results["valid_bbox_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["valid_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 5))

    # Plot loss
    plt.subplot(1, 4, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="valid_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid()

    # Plot accuracy
    plt.subplot(1, 4, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="valid_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid()

    # Plot cls loss
    plt.subplot(1, 4, 3)
    plt.plot(epochs, cls_loss, label="train_cls_loss")
    plt.plot(epochs, test_cls_loss, label="valid_cls_loss")
    plt.title("Cls Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid()

    # Plot bbox loss
    plt.subplot(1, 4, 4)
    plt.plot(epochs, bbox_loss, label="train_bbox_loss")
    plt.plot(epochs, test_bbox_loss, label="valid_bbox_loss")
    plt.title("Bbox Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid()

    if save_fig:
        target_dir = os.path.join(config.PARENT_DIR, "results", config.DATASET_NAME, config.TIMESTAMP, "pred")
        save_plot(target_dir, "loss_acc_curves.png")

def create_confusion_matrix(model, test_loader, class_names, device, save_fig=False):

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
  # create a figure
  plt.figure(figsize=(10, 10))
  plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  plt.xticks(np.arange(len(class_names)), class_names, rotation='vertical')
  plt.yticks(np.arange(len(class_names)), class_names)
  if save_fig:
    target_dir = os.path.join(config.PARENT_DIR, "results", config.DATASET_NAME, config.TIMESTAMP, "pred")
    save_plot(target_dir, "conf_mat.png")

def pred_and_plot_img(model,
                      img_path,
                      class_names,
                      img_size,
                      transform,
                      device):
    try:
        ground_truth_label = img_path.split("/")[-2].split("-")[-1]
    except:
        ground_truth_label = None

    img = Image.open(img_path)
    img = img.resize(size=img_size)

    if transform is not None:
        img_transform = transform
    else:
        img_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=config.MEAN, std=config.STD)
        ])

    model.to(device)

    model.eval()
    
    with torch.inference_mode():
        img_tensor = img_transform(img).unsqueeze(0)
        img_tensor = img_tensor.to(device)
        prediction = model(img_tensor)

    classPred = prediction[0]
    bboxPred = prediction[1]

    (startX, startY, endX, endY) = bboxPred[0]

    # determine the class label with the largest predicted
	# probability
    class_label = torch.argmax(classPred).item()
    label = class_names[class_label]

    # scale the bounding box coordinates
    startX = int(startX * img.size[0])
    startY = int(startY * img.size[1])
    endX = int(endX * img.size[0])
    endY = int(endY * img.size[1])

    # convert PIL image to OpenCV
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # draw the bounding box
    if ground_truth_label is not None:
        box_color = (0, 255, 0) if str(label.lower()) == ground_truth_label.lower() else (0, 0, 255)
    else:
        box_color = (0, 255, 0)
    cv2.rectangle(image, (startX, startY), (endX, endY), box_color, 2)
    cv2.putText(image, str(label), (startX, startY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # show the output image 
    # cv2.imwrite("output_test_pred.jpg", image)

    # convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image, str(label), ground_truth_label

def visualize_dataset(dataloader, save_fig=False):
    # Set figure size
    plt.figure(figsize=(20, 20)) 

    # Set subplot parameters
    plt.subplots(nrows=3, ncols=3, figsize=(7, 7))

    # # Visualize random 9 images in 3x3 grid
    for i in range(9):
        plt.subplot(3, 3, i+1)
        idx = random.randint(0, len(dataloader.dataset))
        image, annotation_cls, annotation_bbox = dataloader.dataset[idx][0].permute(1, 2, 0).numpy(), dataloader.dataset[idx][1], dataloader.dataset[idx][2]
        # image
        image = image * config.STD + config.MEAN
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
        # plt.imshow(image)
        # annotation
        label = annotation_cls.numpy()
        bbox = annotation_bbox.numpy()

        # scale the bounding box coordinates
        startX = int(bbox[0] * image.shape[1])
        startY = int(bbox[1] * image.shape[0])
        endX = int(bbox[2] * image.shape[1])
        endY = int(bbox[3] * image.shape[0])

        # convert PIL image to OpenCV
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # draw the bounding box
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image, str(config.CLASS_NAMES[int(label)]), (startX, startY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.title(f"Label: {config.CLASS_NAMES[int(label)]}")
        plt.axis("off")

    plt.tight_layout()
    if save_fig:
        target_dir = os.path.join(config.PARENT_DIR, "results", config.DATASET_NAME, config.TIMESTAMP, "pred")
        save_plot(target_dir, "visualize_dataset.png")

def save_plot(target_dir: str, filename: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(target_dir, filename))

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

  model_save_path = os.path.join(target_dir_path, model_name)

  # Save the model state_dict()
  # print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)