import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNet
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
import config
import cv2
import pandas as pd
import numpy as np

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


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loss_loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loss_loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loss_loop.set_postfix(loss=loss.item())


def main():

    # Dataset prep
    # Get the class label csv file
    classes = pd.read_csv(DATASET_DIR + '/class_dict.csv', index_col=0)

    # There are total 32 classes
    n_classes = len(classes)
    print(f'Total number of classes: {n_classes}')

    # Create a dictionary mapping for the classes with their colors
    cls2rgb = {cl: list(classes.loc[cl, :]) for cl in classes.index}

    # Index to RGB
    idx2rgb = {idx: np.array(rgb)
               for idx, (cl, rgb) in enumerate(cls2rgb.items())}

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
    loss_fn = nn.CrossEntropyLoss()  # for multi class, use cross entropy
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        dataset_dir=DATASET_DIR,
        batch_size=BATCH_SIZE,
        train_transform=train_transforms,
        val_transform=val_transforms,
        pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model)

    # for Automatic Mixed Precision (AMP) training
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, folder="saved_models",
                        filename="my_checkpoint.pth.tar")

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE)


if __name__ == "__main__":
    main()
