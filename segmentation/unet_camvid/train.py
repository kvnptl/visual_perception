import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNet
from loss_fn import FocalLoss, DiceLoss, TverskyLoss, IoULoss, DiceFocalLoss, DiceCrossEntropyLoss, IoUCrossEntropyLoss, IoUFocalLoss, TverskyCrossEntropyLoss
import utils
import dataloader
import config
import engine
import cv2
import pandas as pd
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from torchinfo import summary
from timeit import default_timer as timer
import argparse

# Select device
parser = argparse.ArgumentParser(description="Choose device index")
"""
NOTE:
# In argparse, arguments that start with a hyphen (-) or double hyphen (--) are optional and can be specified in any order. These are often used for flags or to specify parameters with default values.
# Arguments without a hyphen are positional arguments. They are required and must be provided in the same order they were added.
"""
parser.add_argument('-deviceIndex', metavar='DeviceIndex',
                    type=int, help='the device index to use (0 or 1)', default=0)
args = parser.parse_args()

DEVICE = f"cuda:{args.deviceIndex}" if torch.cuda.is_available() else "cpu"

# Hyperparameters
LEARNING_RATE = config.LEARNING_RATE
BATCH_SIZE = config.BATCH_SIZE
NUM_EPOCHS = config.NUM_EPOCHS
NUM_WORKERS = config.NUM_WORKERS
IMAGE_HEIGHT = config.IMAGE_HEIGHT
IMAGE_WIDTH = config.IMAGE_WIDTH
PIN_MEMORY = config.PIN_MEMORY
LOAD_MODEL = config.LOAD_MODEL
DATASET_DIR = config.DATASET_DIR
LOAD_MODEL_FILE = "/home/kpatel2s/work/kpatel2s_datasets/carvana_dataset/my_checkpoint.pth.tar"

# Model tracking using Tensorboard


# def create_write(experiment_name: str,
#                  model_name: str,
#                  extra: str = None):

#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

#     if extra:
#         log_dir = os.path.join(config.PARENT_DIR, "results",
#                                config.DATASET_NAME, "runs", config.TIMESTAMP, extra)
#     else:
#         log_dir = os.path.join(config.PARENT_DIR, "results",
#                                config.DATASET_NAME, "runs", config.TIMESTAMP)

#     print(f"[INFO] Created SummaryWriter directory: {log_dir}")

#     return SummaryWriter(log_dir=log_dir)


def main():

    # Start time
    start_time = timer()

    # Dataset prep
    # Get the class label csv file
    classes = pd.read_csv(DATASET_DIR + '/class_dict.csv', index_col=0)
    cls2rgb = {cl:list(classes.loc[cl, :]) for cl in classes.index}
    map_class_to_rgb = {i: rgb for i, rgb in enumerate(cls2rgb.values())} # a new dictionary with integer keys

    # There are total 32 classes
    n_classes = len(classes)
    print(f'[INFO] Total number of classes: {n_classes}')

    # Transforms
    train_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    # Test
    test_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    # Set seed
    utils.set_seed(seed=config.SEED)

    model = UNet(in_channels=3, out_channels=n_classes).to(DEVICE)

    # Get the summary of the model
    if config.PRINT_MODEL_SUMMARY:

        x = torch.randn((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)).to(DEVICE)
        summary(model=model,
                input_size=x.shape,
                col_names=["input_size", "output_size",
                           "num_params", "trainable"],
                depth=1,
                row_settings=["var_names"])

    loss_fn = IoULoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # scheduler = optim.lr_scheduler.StepLR(
    #     optimizer, step_size=NUM_EPOCHS/4, gamma=0.1)

    train_dataloader = dataloader.get_loaders(
        dataset_dir=DATASET_DIR,
        batch_size=BATCH_SIZE,
        transform=train_transforms,
        set_type="train",
        pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS
    )

    val_dataloader = dataloader.get_loaders(
        dataset_dir=DATASET_DIR,
        batch_size=BATCH_SIZE,
        transform=val_transforms,
        set_type="val",
        pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS
    )

    test_dataloader = dataloader.get_loaders(
        dataset_dir=DATASET_DIR,
        batch_size=1,
        transform=test_transforms,
        set_type="test",
        pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS
    )

    if LOAD_MODEL:
        utils.load_checkpoint(torch.load(LOAD_MODEL_FILE), model)

    # model_writer = create_write(experiment_name="unet_camvid",
    #                             model_name=f"model_epoch_{NUM_EPOCHS}",)

    # For Automatic Mixed Precision (AMP) training
    scaler = torch.cuda.amp.GradScaler()
    results = engine.train(model=model,
                           train_dataloader=train_dataloader,
                           val_dataloader=val_dataloader,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           epochs=NUM_EPOCHS,
                           device=DEVICE,
                           writer=None,
                           scaler=scaler,
                           scheduler=scheduler if 'scheduler' in globals() else None, # Check if scheduler exists as a variable
                           map_class_to_rgb=map_class_to_rgb,
                           save_model=True)

    # Save model
    model_save_path = os.path.join(
        config.PARENT_DIR, "results", config.DATASET_NAME, config.TIMESTAMP, "model")
    utils.save_model(model=model, target_dir=model_save_path,
                     model_name=f"last_epoch_{NUM_EPOCHS}_{config.MODEL_LOG}.pth")
    print(f"[INFO] Last epoch Model saved to: {model_save_path}")

    # Evaluation
    utils.plot_loss_curve(results=results, save_fig=True)

    # End time
    end_time = timer()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"{'****' * 10}")
    print(
        f"[INFO] Time elapsed: {int(hours):02}:{int(minutes):02}:{seconds:05.2f}")
    print(f"{'****' * 10}")

    if config.LOAD_MODEL:
        print(f"[INFO] Loading model...")
        # load the model weights
        weights_file = config.MODEL_PATH
        model.load_state_dict(torch.load(weights_file))

    # Inference on test set and save the results
    utils.save_predictions_as_imgs(
        test_dataloader, model, num_imgs=len(test_dataloader.dataset), set_type="test", map_class_to_rgb=map_class_to_rgb, device=DEVICE)


if __name__ == "__main__":
    main()
