
# Custom object detector

# Reference: 06 transfer learning


# ### Import libraries
import torch
import torchvision

print(f"torch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")

import os
import numpy as np
import random
import matplotlib.pyplot as plt
from torchinfo import summary
import cv2
from typing import List, Tuple
from PIL import Image
import imutils
from pathlib import Path

from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter

# Custom imports
import model
import engine
from utils import (
    set_seed, 
    plot_loss_curves, 
    create_confusion_matrix, 
    save_model, 
    pred_and_plot_img, 
    crawl_through_dir, 
    visualize_dataset)
import config
import dataset

# ### Load hyperparameters

NUM_WORKERS = config.NUM_WORKERS
BATCH_SIZE = config.BATCH_SIZE
PIN_MEMORY = config.PIN_MEMORY

num_classes = config.NUM_CLASSES

IMAGE_SIZE = config.IMAGE_SIZE

EPOCHS = config.EPOCHS
LR_RATE = config.LR_RATE

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ### Load the dataset


# Setup path to data folder
data_path = Path(config.DATASET)

# image files paths and annotations
images_path = os.path.join(data_path, "images")
annotations_path = os.path.join(data_path, "yolo", "annotations")

# ### Create data loaders

train_dataloader = dataset.create_dataloader(
    images_path=images_path,
    annotations_path=annotations_path,
    subset="train",
    batch_size=BATCH_SIZE,
    pin_memory=PIN_MEMORY,
    transforms=None,
    num_workers=NUM_WORKERS
)

valid_dataloader = dataset.create_dataloader(
    images_path=images_path,
    annotations_path=annotations_path,
    subset="valid",
    batch_size=BATCH_SIZE,
    pin_memory=PIN_MEMORY,
    transforms=None,
    num_workers=NUM_WORKERS
)

test_dataloader = dataset.create_dataloader(
    images_path=images_path,
    annotations_path=annotations_path,
    subset="test",
    batch_size=1,
    pin_memory=PIN_MEMORY,
    transforms=None,
    num_workers=NUM_WORKERS
)

print(f"class_names: {config.CLASS_NAMES}")

# ### Visualize samples from the dataset

visualize_dataset(train_dataloader)


# ### Setup the model


# Create the network
weights = torchvision.models.ResNet50_Weights.DEFAULT
basemodel = torchvision.models.resnet50(weights=weights)

model = model.ObjectDetector(basemodel, num_classes)


# #### Get the model summary

# #### Freeze ResNet50 weights (backbone)
for param in model.backbone.parameters():
    param.requires_grad = False

x = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE))    

summary(model=model, 
        input_size=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])

# NOTE: all ResNet50 layers are not trainable (False) and only 
# classification and bbox regression layers are trainable (True)


# ### Loss function and optimizer


# Two loss functions
classLoss_function = torch.nn.CrossEntropyLoss()
bboxLoss_function = torch.nn.MSELoss()
loss_fn = (classLoss_function, bboxLoss_function)

# Optimizer
optimizer = torch.optim.Adam(params=model.parameters(),
                            lr=LR_RATE)


# ### Model tracking using Tensorboard

def create_write(experiment_name: str,
                 model_name: str,
                 extra: str=None):
    
    from datetime import datetime
    import os

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    if extra:
        los_dir = os.path.join("runs", experiment_name, timestamp, model_name, extra)
    else:
        los_dir = os.path.join("runs", experiment_name, timestamp, model_name)

    print(f"[INFO] Created SummaryWriter directory: {los_dir}")

    return SummaryWriter(log_dir=los_dir)


# ### Train the model

set_seed(seed=config.SEED)

from timeit import default_timer as timer
start_time = timer()

model_writer = create_write(experiment_name="standford_dogs_mini_10",
                           model_name=f"model_epoch_{EPOCHS}",)

results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=valid_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=EPOCHS,
                       device=device,
                       writer=model_writer)

end_time = timer()
print(f"Time elapsed: {end_time - start_time:.3f} seconds")

# Save model
save_model(model=model, target_dir="models", model_name=f"model_{EPOCHS}.pth")


# Evaluation
plot_loss_curves(results=results, save_fig=True)


create_confusion_matrix(model=model,
                        test_loader=test_dataloader,
                        class_names=config.CLASS_NAMES,
                        device=device,
                        save_fig=True)


# ### Inference on images


if config.LOAD_MODEL:
    print("[INFO] Loading model...")
    # load the model weights
    weights_file = config.MODEL_PATH
    model.load_state_dict(torch.load(weights_file))


test_dir = "/home/kpatel2s/kpatel2s/object_detection/custom_object_detector/dataset/standford_dogs_mini_10/images"

num_imgs = 9
test_img_path_list = crawl_through_dir(test_dir)
test_img_path_sample = random.sample(test_img_path_list, num_imgs)

# Set figure size
plt.figure(figsize=(20, 20)) 

# Set subplot parameters
plt.subplots(nrows=3, ncols=3, figsize=(12, 12))

for i, test_img in enumerate(test_img_path_sample):
    # plot output images
    plt.subplot(3, 3, i+1)
    image, label, gt_label = pred_and_plot_img(model=model,
                    img_path=test_img,
                    class_names=config.CLASS_NAMES,
                    img_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
                    transform=None,
                    device=device)
    plt.axis("off")
    plt.imshow(image)
    plt.title(f"Pred: {label.lower()} | GT: {gt_label.lower()}")

plt.tight_layout(pad=2.0)
plt.savefig("output_test_pred.jpg")

# ### Benchmark on Test dataset


# # TODO
# 
# - ~~add tensorboard~~
# - add mAP metric from torchmetrics
# - make modular code
# - save best model based on validation loss
# - separate inference script


