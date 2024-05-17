import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNet
import utils
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

# Hyperparameters
LEARNING_RATE = config.LEARNING_RATE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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

    # Dataset prep
    # Get the class label csv file
    classes = pd.read_csv(DATASET_DIR + '/class_dict.csv', index_col=0)

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

    model = UNet(in_channels=3, out_channels=n_classes).to(DEVICE)

    # Get the summary of the model
    if config.PRINT_MODEL_SUMMARY:

        x = torch.randn((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)).to(DEVICE)
        summary(model=model,
                input_size=x.shape,
                col_names=["input_size", "output_size",
                           "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"])

    loss_fn = nn.CrossEntropyLoss()  # for multi class, use cross entropy
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_dataloader, val_dataloader = utils.get_loaders(
        dataset_dir=DATASET_DIR,
        batch_size=BATCH_SIZE,
        train_transform=train_transforms,
        val_transform=val_transforms,
        pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS
    )

    if LOAD_MODEL:
        utils.load_checkpoint(torch.load(LOAD_MODEL_FILE), model)

    # Set seed
    utils.set_seed(seed=config.SEED)

    start_time = timer()

    # model_writer = create_write(experiment_name="unet_camvid",
    #                             model_name=f"model_epoch_{NUM_EPOCHS}",)

    results = engine.train(model=model,
                           train_dataloader=train_dataloader,
                           test_dataloader=val_dataloader,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           epochs=NUM_EPOCHS,
                           device=DEVICE,
                           writer=None,
                           save_model=True)

    end_time = timer()
    print(f"[INFO] Time elapsed: {end_time - start_time:.3f} seconds")

    # Save model
    model_save_path = os.path.join(
        config.PARENT_DIR, "results", config.DATASET_NAME, config.TIMESTAMP, "model")
    utils.save_model(model=model, target_dir=model_save_path,
                     model_name=f"last_epoch_{NUM_EPOCHS}_{config.MODEL_LOG}.pth")
    print(f"[INFO] Last epoch Model saved to: {model_save_path}")

    # Evaluation
    utils.plot_loss_curve(results=results, save_fig=True)

    # TODO: for Automatic Mixed Precision (AMP) training
    scaler = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL:
        print(f"[INFO] Loading model...")
        # load the model weights
        weights_file = config.MODEL_PATH
        model.load_state_dict(torch.load(weights_file))

    # TODO: integrate properly
    # print some examples to a folder
    utils.save_predictions_as_imgs(
        val_dataloader, model, device=DEVICE)


if __name__ == "__main__":
    main()
