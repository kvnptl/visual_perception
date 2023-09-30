
"""

Return train and test dataloaders for a given dataset.

"""


# import the necessary packages
import os
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
from imutils import paths
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
from utils import xywh_to_xyxy, readFile, set_seed

NUM_WORKERS = os.cpu_count()
# specify ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class_names = ['Chihuahua', 
           'Golden_retriever', 
           'Welsh_springer_spaniel', 
           'German_shepherd', 
           'Doberman', 
           'Boxer', 
           'Siberian_husky', 
           'Pug', 
           'Pomeranian', 
           'Cardigan']

class create_dataset(Dataset):
    def __init__(self,
                 images_path: str,
                 annotations_path: str,
                 subset: str="train",
                 transforms=None):
        super().__init__()

        self.subset = subset
        self.transforms = transforms

        # read the image file paths
        images_list = self.crawl_through_dir(images_path)
        annotations_list = self.crawl_through_dir(annotations_path)

        # random shuffle the images and annotations
        data = list(zip(images_list, annotations_list))
        random.shuffle(data)
        images_list, annotations_list = zip(*data)

        # split the dataset into training and testing splits
        image_set, annot_set = self.split_dataset(images_list, annotations_list, val_size=0.10, test_size=0.10)

        self.image_subset = image_set[subset]
        self.annot_subset = annot_set[subset]

        if self.transforms is None:
            # define normalization transforms
            self.transforms = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])

    def __len__(self):
        return len(self.image_subset)
    
    def __getitem__(self, idx):
        image, annot = self.getImageAndAnnotation(idx)

        img_tensor = self.transforms(image)
        annot_tensor = torch.from_numpy(annot)

        # NOTE: the current network is designed to work with only one bounding box per image
        annot_tensor = annot_tensor[0]

        return (img_tensor, annot_tensor)
    
    def getImageAndAnnotation(self, idx):
        # load the image
        img = Image.open(self.image_subset[idx]).convert("RGB")

        # load the annotation
        annots = self.getAnnotation(img, self.annot_subset[idx])

        return img, annots

    def getAnnotation(self, ref_img, annotation):
        
        annotations = []

        # load the contents of the current CSV annotations file
        rows = open(annotation).read().strip().split("\n")
        # loop over the rows
        for row in rows:
            # break the row into the filename, bounding box coordinates,
            # and class label
            row = row.split(" ")
            # convert xywh to xyxy
            row[1:] = xywh_to_xyxy(row[1:])
            # convert label to int
            label = int(row[0])
            (startX, startY, endX, endY) = row[1:]
            # derive the path to the input image, load the image (in
            # OpenCV format), and grab its dimensions
            (orig_w, orig_h) = ref_img.size

            startX = int(float(startX) * orig_w)
            startY = int(float(startY) * orig_h)
            endX = int(float(endX) * orig_w)
            endY = int(float(endY) * orig_h)

            # draw the bounding box on the image
            # cv_img = cv2.cvtColor(np.array(ref_img), cv2.COLOR_RGB2BGR)
            # cv2.rectangle(cv_img, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # cv2.imwrite("test_1.png", cv_img)

            # resize the image to 224x224 using PIL
            ref_img = ref_img.resize((224, 224))
            (new_w, new_h) = ref_img.size

            # scale the bounding box coordinates relative to the spatial
            # dimensions of the input image
            startX = int((startX) * new_w / orig_w)
            startY = int((startY) * new_h / orig_h)
            endX = int((endX) * new_w / orig_w)
            endY = int((endY) * new_h / orig_h)

            # draw the bounding box on the image
            # cv_img = cv2.cvtColor(np.array(ref_img), cv2.COLOR_RGB2BGR)
            # cv2.rectangle(cv_img, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # cv2.imwrite("test_2.png", cv_img)

            # normalize the bounding box coordinates
            startX = startX / new_w
            startY = startY / new_h
            endX = endX / new_w
            endY = endY / new_h

            annotations.append([label, startX, startY, endX, endY])

        # convert to numpy
        annotations = np.array(annotations, dtype="float32")
        return annotations

    def crawl_through_dir(self, dir_path):
        file_paths = []
        for root, directories, files in os.walk(dir_path):
            for filename in files: 
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)
        
        file_paths = sorted(file_paths)
        return file_paths
    
    def split_dataset(self, images_list, annotations_list, val_size=0.10, test_size=0.10):

        train_size = 1 - val_size - test_size
        val_size = train_size + val_size

        # split the image_names into train, valid, eval using the ratio 0.7, 0.15, 0.15 
        self.train_set_imgs, self.valid_set_imgs, self.test_set_imgs = np.split(images_list, [int(train_size*len(images_list)), int(val_size*len(images_list))])

        # same for annotations
        self.train_set_annots, self.valid_set_annots, self.test_set_annots = np.split(annotations_list, [int(train_size*len(annotations_list)), int(val_size*len(annotations_list))])

        image_set = {'train': self.train_set_imgs, 'valid': self.valid_set_imgs, 'test': self.test_set_imgs}
        annot_set = {'train': self.train_set_annots, 'valid': self.valid_set_annots, 'test': self.test_set_annots}

        return image_set, annot_set

def create_dataloader(
    images_path: str,
    annotations_path: str,
    subset: str="train",
    batch_size: int=8,
    pin_memory: bool=True,
    transforms=None,
    num_workers: int=NUM_WORKERS
):
    dataSet = create_dataset(
        images_path=images_path,
        annotations_path=annotations_path,
        subset=subset,
        transforms=transforms
    )

    dataloader = DataLoader(
        dataset=dataSet,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=True if subset == "train" else False
    )

    return dataloader

# Test dataloader
def main():

    # set_seed(8)

    images_path = "/home/kpatel2s/kpatel2s/object_detection/custom_object_detector/dataset/standford_dogs_mini_10/images"
    annotations_path = "/home/kpatel2s/kpatel2s/object_detection/custom_object_detector/dataset/standford_dogs_mini_10/yolo/annotations"

    # create dataloaders
    train_dataloader = create_dataloader(
        images_path=images_path,
        annotations_path=annotations_path,
        subset="train",
        batch_size=8,
        pin_memory=True
    )

    # get one sample from the training dataloader
    image, annotation = train_dataloader.dataset[0]
    print(image.shape)

    image = image.permute(1, 2, 0).numpy()
    image = image * STD + MEAN
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)

    # show the image
    plt.imshow(image)
    plt.savefig("test.png")

    # draw the bounding box
    label = annotation[0].numpy()
    bbox = annotation[1:].numpy()

    # scale the bounding box coordinates
    startX = int(bbox[0] * image.shape[1])
    startY = int(bbox[1] * image.shape[0])
    endX = int(bbox[2] * image.shape[1])
    endY = int(bbox[3] * image.shape[0])

    # convert PIL image to OpenCV
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # draw the bounding box
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.putText(image, str(class_names[int(label)]), (startX, startY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # save the image
    cv2.imwrite("test.png", image)

if __name__ == "__main__":
    main()
    








