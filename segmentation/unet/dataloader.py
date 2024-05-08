import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class CarvanaDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, indices=None):
        self.image_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.all_images = os.listdir(self.image_dir)
        self.indices = indices if indices is not None else range(
            len(self.image_dir))
        self.images = [self.all_images[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(
            self.mask_dir, self.images[index].replace('.jpg', '_mask.gif'))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255] = 1.0

        if self.transform is not None:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]

        return image, mask
