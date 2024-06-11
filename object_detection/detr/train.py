# Import libraries
import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
from torchvision.models import resnet50
from torchinfo import summary
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import os
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import itertools

# Custom imports
import dataloader
import config
import utils
import engine
import model

# Select device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Parameters configuration
DATASET_DIR = config.DATASET_DIR
PARENT_DIR = config.PARENT_DIR

IMAGE_HEIGHT = config.IMAGE_HEIGHT
IMAGE_WIDTH = config.IMAGE_WIDTH

PIN_MEMORY = config.PIN_MEMORY
LEARNING_RATE = config.LEARNING_RATE
BATCH_SIZE = config.BATCH_SIZE
NUM_EPOCHS = config.NUM_EPOCHS
NUM_WORKERS = config.NUM_WORKERS

SEED = config.SEED
PRINT_MODEL_SUMMARY = config.PRINT_MODEL_SUMMARY

# Setup path to data folder
data_path = Path(DATASET_DIR)
image_path = data_path / "images"
label_path = data_path / "labels"
class_names_file = data_path / "class_names.txt"

train_csv_file = data_path / "100examples.csv"
test_csv_file = data_path / "test.csv"

# Download dataset

# Read class names txt file
with open(class_names_file, "r") as f:
    classes = [class_name.strip() for class_name in f.readlines()]

# Create class to index dictionary
class_to_idx = {int(class_name.split(":")[0]): class_name.split(":")[1].strip().strip("'") for i, class_name in enumerate(classes)}

NUM_CLASSES = len(classes)

print(f"Number of classes: {len(classes)}")

# Read train CSV
df = utils.read_csv_file(train_csv_file)

# Create dataloader

# Albumentations transforms
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomCrop(224, 224, p=0.5),
    A.ColorJitter(p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()],
    bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))

test_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()],
    bbox_params=A.BboxParams(format='yolo', label_fields=['labels'])
)

train_dataloader, val_dataloader = dataloader.get_loaders(dataset_dir=train_csv_file,
                                image_dir=image_path,
                                label_dir=label_path,
                                batch_size=BATCH_SIZE,
                                transform=train_transform,
                                set_type="train",
                                num_workers=NUM_WORKERS,
                                pin_memory=True)

test_dataloader = dataloader.get_loaders(dataset_dir=test_csv_file,
                                image_dir=image_path,
                                label_dir=label_path,
                                batch_size=1,
                                transform=test_transform,
                                set_type="test",
                                num_workers=NUM_WORKERS,
                                pin_memory=True)


class DETRModel(nn.Module):
    def __init__(self, num_classes=20, num_queries=100):
        super(DETRModel,self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        # Download pre-trained model
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        self.in_features = self.model.class_embed.in_features
        
        self.model.class_embed = nn.Linear(in_features=self.in_features, out_features=self.num_classes)
        self.model.num_queries = self.num_queries
        
    def forward(self,imgs):
        return self.model(imgs)

# Load model
# net = model.DETRdemo(num_classes=20).to(DEVICE)
net = DETRModel(num_classes=NUM_CLASSES+1).to(DEVICE)

# Summary
if PRINT_MODEL_SUMMARY:
        from torchinfo import summary

        summary(model=net,
                input_size=(1, 3, 224, 224), # (batch_size, channels, height, width)
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"])

# Loss and optimizer
from loss_fn import SetCriterion
from hungarian_matcher import HungarianMatcher

matcher = HungarianMatcher()

weight_dict = weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}
losses = ['labels', 'boxes', 'cardinality']
NULL_CLASS_COEF = 0.5

criterion = SetCriterion(NUM_CLASSES, matcher, weight_dict, eos_coef = NULL_CLASS_COEF, losses=losses).to(DEVICE) # eos_coef is used in the output layer to affect the output corresponding to the absence of an object.

optimizer = torch.optim.Adam(params=net.parameters(),
                             lr=LEARNING_RATE,
                             weight_decay=0.1)

# Train model
utils.set_seeds(SEED)

# Train the model
results = engine.train(model=net,
                      train_dataloader=train_dataloader,
                      val_dataloader=val_dataloader,
                      optimizer=optimizer,
                      loss_fn=criterion,
                      epochs=NUM_EPOCHS,
                      device=DEVICE)

# Plot loss and accuracy curves
utils.plot_loss_curves(results=results, save_fig=True)