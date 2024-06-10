import torch
import torchvision
from pathlib import Path
import matplotlib.pyplot as plt
import os
import requests
import zipfile
from typing import List

import config

def get_color_map() -> List:
    colors = [(51, 51, 153), (153, 255, 153), (153, 153, 255), (255, 255, 153), (255, 153, 255), (153, 255, 255), (204, 102, 102), (102, 204, 102), (102, 102, 204), (204, 204, 102), (204, 102, 204), (102, 204, 204), (153, 51, 51), (51, 153, 51), (255, 153, 153), (153, 153, 51), (153, 51, 153), (51, 153, 153), (229, 0, 0), (0, 229, 0)]

    return colors

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
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
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    if save_fig:
        target_dir = os.path.join(
            config.SELF_DIR, "results", config.DATASET_NAME, config.TIMESTAMP, "plots")
        save_plot(target_dir, "loss_and_acc_curves.png")

def save_plot(target_dir: str, filename: str):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    plt.savefig(os.path.join(target_dir, filename))