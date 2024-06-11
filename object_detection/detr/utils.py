import torch
import torchvision
from pathlib import Path
import matplotlib.pyplot as plt
import os
import requests
import zipfile
from typing import List
import pandas as pd

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
            {"loss1": [...],
             "accuracy1": [...],
             "loss2": [...],
             "accuracy2": [...]}
    """

    # Read results dictionary keys
    keys = [k for k in results.keys()]

    loss = results[keys[0]]
    test_loss = results[keys[2]]

    accuracy = results[keys[1]]
    test_accuracy = results[keys[3]]

    epochs = range(len(results[keys[0]]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label=keys[0])
    plt.plot(epochs, test_loss, label=keys[2])
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label=keys[1])
    plt.plot(epochs, test_accuracy, label=keys[3])
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

def read_csv_file(file_path):
    # Read the first line of the file
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()

    # Split the first line and check the extension of the first entry
    first_entry = first_line.split(',')[0]
    _, ext = os.path.splitext(first_entry)

    # If the first entry has an extension, it's likely a filename and the CSV file has no header
    if ext:
        df = pd.read_csv(file_path, header=None, names=["img", "label"])
    else:
        # Otherwise, assume the CSV file has a header
        df = pd.read_csv(file_path, skiprows=1, names=["img", "label"])

    return df