
### 2.1 Create Dataset and Dataloaders (script mode)

"""
Contains functionality for creating PyTorch DataLoader's for 
image classification data.
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch

import os
from pathlib import Path
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

colors = [(51, 51, 153), (153, 255, 153), (153, 153, 255), (255, 255, 153), (255, 153, 255), (153, 255, 255), (204, 102, 102), (102, 204, 102), (102, 102, 204), (204, 204, 102), (204, 102, 204), (102, 204, 204), (153, 51, 51), (51, 153, 51), (255, 153, 153), (153, 153, 51), (153, 51, 153), (51, 153, 153), (229, 0, 0), (0, 229, 0)]

def get_loaders(
        dataset_dir,
        image_dir,
        label_dir,
        batch_size=16,
        transform=None,
        set_type="train",
        num_workers=4,
        pin_memory=True,
        split_ratio=0.9  # parameter for split ratio
):
    df = pd.read_csv(dataset_dir, header=None, names=["img", "label"])
    
    full_dataset = PascalVOCDataset(df, image_dir, label_dir, transform=transform)
    
    if set_type == "train":
        # Calculate the lengths of train/val datasets
        train_len = int(split_ratio * len(full_dataset))
        val_len = len(full_dataset) - train_len

        # Split the dataset
        train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

        # Create the dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(set_type == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=(set_type == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory
        )

        return train_dataloader, val_dataloader
    
    else:
        test_dataloader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            shuffle=(set_type == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        return test_dataloader

class PascalVOCDataset(Dataset):
    def __init__(self, df, image_dir, label_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.df.iloc[idx, 0])
        label_path = os.path.join(self.label_dir, self.df.iloc[idx, 1])

        # image = Image.open(img_path)
        # image = image.convert("RGB")
        # image = np.array(image)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        with open(label_path) as file:
            labels = file.readlines()
        
        bboxes = []
        class_labels = []
        for label in labels:
            parts = label.strip().split()
            class_labels.append(int(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            bbox_width = float(parts[3])
            bbox_height = float(parts[4])
            bboxes.append([x_center, y_center, bbox_width, bbox_height])
        
        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, labels=class_labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['labels']
        
        target = {}
        target['boxes'] = torch.tensor(bboxes, dtype=torch.float32)
        target['labels'] = torch.tensor(class_labels, dtype=torch.int64)
        
        return image, target

def class_names(class_names_file):
    # Read class names txt file
    with open(class_names_file, "r") as f:
        classes = [class_name.strip() for class_name in f.readlines()]

    # Create class to index dictionary
    class_to_idx = {int(class_name.split(":")[0]): class_name.split(":")[1].strip().strip("'") for i, class_name in enumerate(classes)}

    return classes, class_to_idx

def main():

    # Setup path to data folder
    data_path = Path("/home/kpatel2s/work/kpatel2s_datasets/pascal_voc_dataset")
    image_path = data_path / "images"
    label_path = data_path / "labels"
    class_names_file = data_path / "class_names.txt"

    train_csv_file = data_path / "100examples.csv"
    test_csv_file = data_path / "test.csv"

    classes, class_to_idx = class_names(class_names_file)

    NUM_WORKERS = 4
    BATCH_SIZE = 1

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

    train_dataloader, val_dataloader = get_loaders(dataset_dir=train_csv_file,
                                    image_dir=image_path,
                                    label_dir=label_path,
                                    batch_size=BATCH_SIZE,
                                    transform=train_transform,
                                    set_type="train",
                                    num_workers=NUM_WORKERS,
                                    pin_memory=True)

    test_dataloader = get_loaders(dataset_dir=test_csv_file,
                                   image_dir=image_path,
                                   label_dir=label_path,
                                   batch_size=1,
                                   transform=test_transform,
                                   set_type="test",
                                   num_workers=NUM_WORKERS,
                                   pin_memory=True)
    
    print(f"Train samples: {len(train_dataloader)}")
    print(f"Val samples: {len(val_dataloader)}")
    print(f"Test samples: {len(test_dataloader)}")

    # Take one random image from test samples and draw bboxes
    image, target = next(iter(train_dataloader))
    
    # Convert the image tensor to a numpy array and denormalize it
    image = image[0].permute(1, 2, 0).detach().cpu().numpy()
    image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    bboxes = target['boxes'].numpy()[0]
    labels = target['labels'].numpy()[0]

    height, width = image.shape[:2]

    # Draw bboxes
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        x, y, w, h = bbox

        x *= width
        y *= height
        w *= width
        h *= height

        left = int(x - w / 2)
        top = int(y - h / 2)

        cv2.rectangle(image, (int(left), int(top)), (int(left + w), int(top + h)), colors[int(label)], 2)

        text = f'{class_to_idx[int(label)]}'
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (left, top - text_size[1] - 5), (left + text_size[0] + 5, top), colors[int(label)], -1)
        cv2.putText(image, text, (left + 2, top - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

    # Save image
    cv2.imwrite("test.png", image)

if __name__ == "__main__":
    main()
