import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
from torchinfo import summary
import data_setup, engine, utils, config, model
from pathlib import Path

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
DATASET_DIR = config.DATASET_DIR
DATASET_NAME = config.DATASET_NAME
IMG_SIZE = config.IMG_SIZE
NUM_EPOCHS = config.NUM_EPOCHS
LEARNING_RATE = config.LEARNING_RATE
NUM_WORKERS = config.NUM_WORKERS
BATCH_SIZE = config.BATCH_SIZE
PATCH_SIZE = config.PATCH_SIZE
INPUT_CHANNELS = config.INPUT_CHANNELS
EMBEDDING_DIM = config.EMBEDDING_DIM
NUM_HEADS = config.NUM_HEADS
NUM_TRANSFORMER_LAYERS = config.NUM_TRANSFORMER_LAYERS
MLP_SIZE = config.MLP_SIZE
PRINT_MODEL_SUMMARY = config.PRINT_MODEL_SUMMARY

# Download dataset
utils.download_dataset()

# Prepare dataset
data_path = Path(DATASET_DIR)
image_path = data_path / DATASET_NAME

# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

# Data transforms
manual_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Setup Dataloaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=manual_transform,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS)

# Create ViT model
model = model.ViT(
    img_size=IMG_SIZE,
    in_channels=INPUT_CHANNELS,
    patch_size=PATCH_SIZE,
    num_transformer_layers=NUM_TRANSFORMER_LAYERS,
    embedding_dim=EMBEDDING_DIM,
    mlp_size=MLP_SIZE,
    num_heads=NUM_HEADS,
    attn_dropout=0,
    mlp_dropout=0.1,
    embedding_dropout=0.1,
    num_classes=len(class_names)
)

if PRINT_MODEL_SUMMARY:
    summary(model=model,
            input_size=(1, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            row_settings=["var_names"])

# Loss and Optimizer
loss_fn = nn.CrossEntropyLoss() # the paper didn't mention a loss function
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=LEARNING_RATE,
                             weight_decay=0.1)


# Set seeds
utils.set_seeds()

# Train the model
results = engine.train(model=model,
                      train_dataloader=train_dataloader,
                      test_dataloader=test_dataloader,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      epochs=NUM_EPOCHS,
                      device=device)

# Plot loss and accuracy curves
utils.plot_loss_curves(results=results, save_fig=True)