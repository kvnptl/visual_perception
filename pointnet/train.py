
# # PointNet

# This is an implementation of [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593) using PyTorch.
# 
# Reference: [Deep Learning on Point clouds - Towards Data Science](https://towardsdatascience.com/deep-learning-on-point-clouds-implementing-pointnet-in-google-colab-1fd65cd3a263)
# 

# ## Imports

import numpy as np
import math
import random
import os
import torch
import scipy.spatial.distance
from pathlib import Path
import pandas as pd
import zipfile
import requests
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import plotly.graph_objects as go
import plotly.express as px

import model
from utils import visualize_rotate, pcshow, read_off, set_seed
import dataset
import config

set_seed(8)

# Download the [dataset](http://3dvision.princeton.edu/projects/2014/3DShapeNets/) directly to the Google Colab Runtime. It comprises 10 categories, 3,991 models for training and 908 for testing.

# Setup path to data folder
# data_path = Path("dataset/")
data_path = Path(config.DATASET)

# If the image folder doesn't exist, download it and prepare it... 
if data_path.is_dir():
    print(f"{data_path} directory exists.")
else:
    print(f"Did not find {data_path} directory, creating one...")
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Download ModelNet10
    with open(data_path / "ModelNet10.zip", "wb") as f:
        print("Downloading ModelNet10 data...")
        request = requests.get("http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip")
        f.write(request.content)

    # Unzip ModelNet10
    with zipfile.ZipFile(data_path / "ModelNet10.zip", "r") as zip_ref:
        print("Unzipping ModelNet10 data...") 
        zip_ref.extractall(data_path)

    # Remove zip file
    print("Removing zip file...")
    os.remove(data_path / "ModelNet10.zip")


# path = Path("dataset/ModelNet10")
path = Path(config.DATASET+ '/' + config.DATASET_NAME)


folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path/dir)]
classes = {folder: i for i, folder in enumerate(folders)};
print(f"Classes: {classes}")


# This dataset consists of **.off** files that contain meshes represented by *vertices* and *triangular faces*.
# 
# We will need a function to read this type of files:


with open(path/"bed/train/bed_0001.off", 'r') as f:
  verts, faces = read_off(f)

i,j,k = np.array(faces).T
x,y,z = np.array(verts).T

print(f"There are {len(x)} points in the dataset.")

# Don't be scared of this function. It's just to display animated rotation of meshes and point clouds.
# visualize_rotate([go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.50, i=i,j=j,k=k)]).show()

# This mesh definitely looks like a bed.
# visualize_rotate([go.Scatter3d(x=x, y=y, z=z,
#                                    mode='markers')]).show()

# Unfortunately, that's not the case for its vertices. It would be difficult for PointNet to classify point clouds like this one.

# First things first, let's write a function to accurately visualize point clouds so we could see vertices better.
# pcshow(x,y,z)

pointcloud = dataset.PointSampler(3000)((verts, faces))
# pcshow(*pointcloud.T)

# This pointcloud looks much more like a bed!


# ### Normalize
# Unit sphere
norm_pointcloud = dataset.Normalize()(pointcloud)
# pcshow(*norm_pointcloud.T)
# Notice that axis limits have changed.


rot_pointcloud = dataset.RandRotation_z()(norm_pointcloud)
noisy_rot_pointcloud = dataset.RandomNoise()(rot_pointcloud)

# pcshow(*noisy_rot_pointcloud.T)


dataset.ToTensor()(noisy_rot_pointcloud)

# Transforms for training. 1024 points per cloud as in the paper!
train_transforms = transforms.Compose([
                    dataset.PointSampler(1024),
                    dataset.Normalize(),
                    dataset.RandRotation_z(),
                    dataset.RandomNoise(),
                    dataset.ToTensor()
                    ])

train_ds = dataset.PointCloudData(path, transform=train_transforms)
valid_ds = dataset.PointCloudData(path, valid=True, folder='test', transform=train_transforms)

inv_classes = {i: cat for cat, i in train_ds.classes.items()};
inv_classes

print('Train dataset size: ', len(train_ds))
print('Valid dataset size: ', len(valid_ds))
print('Number of classes: ', len(train_ds.classes))
print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())
print('Class: ', inv_classes[train_ds[0]['category']])

# Create dataloader
train_loader = DataLoader(dataset=train_ds, 
                          batch_size=config.BATCH_SIZE,
                          num_workers=config.NUM_WORKERS,
                          pin_memory=config.PIN_MEMORY,
                          shuffle=True)
valid_loader = DataLoader(dataset=valid_ds, 
                          batch_size=config.BATCH_SIZE,
                        #   num_workers=config.NUM_WORKERS,
                          pin_memory=config.PIN_MEMORY, 
                          shuffle=False)


# ## Model

# You can find a pretrained model [here](https://drive.google.com/open?id=1nDG0maaqoTkRkVsOLtUAR9X3kn__LMSL)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

pointnet = model.PointNet(classes=config.NUM_CLASSES)
pointnet.to(device)

optimizer = torch.optim.Adam(pointnet.parameters(), lr=config.LR_RATE)
loss_fn = model.pointnetloss

def train(model, train_loader, val_loader=None,  epochs=config.EPOCHS, save=True):
    for epoch in range(epochs):
        pointnet.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = pointnet(inputs.transpose(1,2))

            loss = loss_fn(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                    print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                        (epoch + 1, i + 1, len(train_loader), running_loss / 10))
                    running_loss = 0.0

        pointnet.eval()
        correct = total = 0

        # validation
        if val_loader:
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, __, __ = pointnet(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100. * correct / total
            print('Valid accuracy: %d %%' % val_acc)

        # save the model
        if save:
            torch.save(pointnet.state_dict(), "save_"+str(epoch)+".pth")

# Train the model
train(pointnet, train_loader, valid_loader,  save=False)