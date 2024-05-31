import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import config
import pandas as pd

# Dataset prep
# Get the class label csv file
classes = pd.read_csv(config.DATASET_DIR + '/class_dict.csv', index_col=0)

# There are total 32 classes
n_classes = len(classes)

# Create a dictionary mapping for the classes with their colors
cls2rgb = {cl: list(classes.loc[cl, :]) for cl in classes.index}

def get_loaders(
    dataset_dir,
    batch_size,
    transform,
    set_type="train",
    num_workers=4,
    pin_memory=True,
):
    input_dataset = CamVidDataset(
        img_dir=dataset_dir + "/" + set_type,
        mask_dir=dataset_dir + "/" + set_type + "_labels",
        transform=transform
    )

    if set_type == "train":
        shuffle_flag = True
    else:
        shuffle_flag = False

    data_loader = DataLoader(
        input_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle_flag,
    )

    return data_loader

def adjust_mask(mask, flat=False):
    semantic_map = []
    for color in list(cls2rgb.values()):
        # Check if the mask is equal to the color, return True or False, same shape as mask
        equality = np.equal(mask, color)
        # Check if all the values in the array are True, shape WxH (no channels)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)  # Append the class map

    semantic_map = np.stack(semantic_map, axis=-
                            1).astype(np.uint8)  # Shape WxHx32

    if flat:
        semantic_map = np.reshape(semantic_map, (-1, 256*256))

    return np.float32(semantic_map)


class CamVidDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.image_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(self.image_dir)
        # self.indices = indices if indices is not None else range(
        #     len(self.image_dir))
        # self.images = [self.all_images[i] for i in self.indices]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(
            self.mask_dir, self.images[index].replace('.png', '_L.png'))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGB"))
        mask = adjust_mask(mask, )

        if self.transform is not None:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]

        # Convert one-hot encoded mask to ckass indices
        if mask.shape[-1] == 3 or mask.shape[-1] == 32:
            mask = torch.argmax(mask, dim=-1)

        return image, mask
