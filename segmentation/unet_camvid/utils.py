import torch
import torchvision
from dataloader import CamVidDataset
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import os
import numpy as np
import matplotlib.pyplot as plt
import dataloader
import random
import config


def set_seed(seed=8):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_checkpoint(state, folder="checkpoints/", filename="mt_checkpoint.pth.tar"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    print("=> Saving checkpoint")
    torch.save(state, os.path.join(folder, filename))


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    dataset_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_dataset = CamVidDataset(
        img_dir=dataset_dir + "/train",
        mask_dir=dataset_dir + "/train_labels",
        transform=train_transform
    )

    val_dataset = CamVidDataset(
        img_dir=dataset_dir + "/val",
        mask_dir=dataset_dir + "/val_labels",
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def pixel_accuracy(outputs, masks):
    # dim=1 because outputs is of shape (batch_size, num_classes, height, width) and preds is of shape (batch_size, height, width), so we are taking the max along the num_classes dimension
    _, preds = torch.max(outputs, dim=1)
    # element-wise comparison
    correct = (preds == masks).float()
    # sum along the height and width
    acc = correct.sum() / correct.numel()
    return acc.item()


def dice_score_fn(outputs, masks, smooth=1e-8):
    num_classes = outputs.shape[1]
    _, preds = torch.max(outputs, dim=1)
    dice = 0

    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        true_cls = (masks == cls).float()
        intersection = (pred_cls * true_cls).sum()
        union = pred_cls.sum() + true_cls.sum()
        dice += (2. * intersection + smooth) / (union + smooth)

    dice /= num_classes
    return dice.item()


def iou_score_fn(outputs, masks, smooth=1e-8):
    num_classes = outputs.shape[1]
    _, preds = torch.max(outputs, dim=1)
    iou = 0

    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        true_cls = (masks == cls).float()
        intersection = (pred_cls * true_cls).sum()
        union = pred_cls.sum() + true_cls.sum() - intersection
        iou += (intersection + smooth) / (union + smooth)

    iou /= num_classes
    return iou.item()


def check_accuracy(loader, model, device="cuda"):
    num_pixels = 0
    dice_score = 0
    iou_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            preds = torch.sigmoid(model(x))
            num_pixels += pixel_accuracy(preds, y)
            dice_score += dice_score_fn(preds, y)
            iou_score += iou(preds, y)

    # Print accuracies
    avg_pixel_acc = num_pixels / len(loader)
    avg_dice = dice_score / len(loader)
    avg_iou = iou_score / len(loader)

    print(f"Pixel Accuracy: {avg_pixel_acc}")
    print(f"Dice Score: {avg_dice}")
    print(f"IoU Score: {avg_iou}")

    model.train()


def visualize(idx, image, pred, ground_truth, folder="saved_images/"):
    # Create a folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Convert mask_cls to RGB
    pred_rgb = mask_cls_to_rgb(pred)
    ground_truth_rgb = mask_cls_to_rgb(ground_truth)

    # Create a figure
    plt.figure(figsize=(15, 5))

    # Display the image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Input Image")

    # Display the predicted mask
    plt.subplot(1, 3, 2)
    plt.imshow(pred_rgb)
    plt.axis("off")
    plt.title("Predicted Mask")

    # Display the ground truth mask
    plt.subplot(1, 3, 3)
    plt.imshow(ground_truth_rgb)
    plt.axis("off")
    plt.title("Ground Truth Mask")

    # Save the figure
    file_name = os.path.join(folder, f"_image_{idx}.png")
    plt.savefig(file_name)


def save_predictions_as_imgs(loader, model, device="cuda"):

    target_dir = os.path.join(config.PARENT_DIR, "results",
                              config.DATASET_NAME, "saved_images")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
            preds = torch.argmax(preds, dim=1)
            x_cpu = x.cpu()
            preds_cpu = preds.cpu()
            y_cpu = y.cpu()
            visualize(idx, x_cpu[0].permute(1, 2, 0),
                      preds_cpu[0], y_cpu[0], folder=target_dir)

    model.train()


# Map the idx to RGB
def map_class_to_rgb(p):

    # Index to RGB
    idx2rgb = {idx: np.array(rgb)
               for idx, (cl, rgb) in enumerate(dataloader.cls2rgb.items())}
    return idx2rgb[p[0]]


def convert_mask_to_rgb(new_mask):
    # Get the index of the maximum value along the last axis
    mask_indices = np.argmax(new_mask, axis=-1)

    # Expand the dimensions of the array
    expanded_indices = np.expand_dims(mask_indices, axis=-1)

    # Apply the function `map_class_to_rgb` to each 1-D slice along the last axis
    rgb_mask = np.apply_along_axis(map_class_to_rgb, -1, expanded_indices)

    return rgb_mask


def mask_cls_to_rgb(mask):
    # Expand the dimensions of the array
    expanded_indices = np.expand_dims(mask, axis=-1)

    # Apply the function `map_class_to_rgb` to each 1-D slice along the last axis
    rgb_mask = np.apply_along_axis(map_class_to_rgb, -1, expanded_indices)

    return rgb_mask


def save_model(model, target_dir, model_name):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"), "model_name should end with '.pt' or '.pth'"

    model_save_path = os.path.join(target_dir, model_name)

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)


def plot_loss_curve(results, save_fig=False):
    """
    Plots the loss and accuracy curves of a results dictionary.
    """

    train_loss = results["train_loss"]
    train_acc = results["train_acc"]
    train_dice_score = results["train_dice_score"]
    train_iou_score = results["train_iou_score"]

    val_loss = results["val_loss"]
    val_acc = results["val_acc"]
    val_dice_score = results["val_dice_score"]
    val_iou_score = results["val_iou_score"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 5))

    # Plot loss
    plt.subplot(1, 4, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid()

    # Plot accuracy
    plt.subplot(1, 4, 2)
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid()

    # Plot dice score
    plt.subplot(1, 4, 3)
    plt.plot(epochs, train_dice_score, label="train_dice_score")
    plt.plot(epochs, val_dice_score, label="val_dice_score")
    plt.title("Dice Score")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid()

    # Plot iou score
    plt.subplot(1, 4, 4)
    plt.plot(epochs, train_iou_score, label="train_iou_score")
    plt.plot(epochs, val_iou_score, label="val_iou_score")
    plt.title("IoU Score")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid()

    if save_fig:
        target_dir = os.path.join(
            config.PARENT_DIR, "results", config.DATASET_NAME, config.TIMESTAMP, "plots")
        save_plot(target_dir, "loss_and_acc_curves.png")


def save_plot(target_dir: str, filename: str):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    plt.savefig(os.path.join(target_dir, filename))
