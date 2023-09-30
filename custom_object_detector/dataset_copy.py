
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

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
from utils import xywh_to_xyxy, readFile, set_seed

NUM_WORKERS = os.cpu_count()
# specify ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def extractData(
            images_file_path: str,
            annotations_path: str,
):
    # create empty lists
    images = []
    labels = []
    bboxes = []
    imagePaths = []

    # read the image file paths
    image_file_paths_list = readFile(images_file_path)

    # print dataset statistics
    print(f"Total dataset size: {len(image_file_paths_list)}")

    # loop over all CSV files in the annotations directory
    for txtPath in tqdm(paths.list_files(annotations_path, validExts=(".txt"))):
        # get the file name
        basename = os.path.basename(txtPath)
        basename_no_ext = os.path.splitext(basename)[0]

        # get image path
        for filename in image_file_paths_list:
            if basename_no_ext in filename:
                imagePath = filename.strip()
                break
        
        if imagePath is None:
            print("imagePath is None")
            continue
            
        # load the contents of the current CSV annotations file
        rows = open(txtPath).read().strip().split("\n")
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
            # imagePath = os.path.sep.join([IMAGES_PATH, label,
            # 	filename])
            image = cv2.imread(imagePath)
            # imagePIL = Image.open(imagePath)
            (orig_h, orig_w) = image.shape[:2]
            # (orig_w, orig_h) = imagePIL.size

            # preprocess the image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            (new_h, new_w) = image.shape[:2]
            # image = imagePIL.convert("RGB")
            # image = image.resize((224, 224))
            # (new_w, new_h) = image.size

            # scale the bounding box coordinates relative to the spatial
            # dimensions of the input image
            startX = float(startX) * new_w / orig_w
            startY = float(startY) * new_h / orig_h
            endX = float(endX) * new_w / orig_w
            endY = float(endY) * new_h / orig_h

            # update our list of data, class labels, bounding boxes, and
            # image paths
            images.append(image)
            labels.append(label)
            bboxes.append((startX, startY, endX, endY))
            imagePaths.append(imagePath)

    # convert the data, class labels, bounding boxes, and image paths to
    # NumPy arrays
    images = np.array(images, dtype="float32")
    labels = np.array(labels)
    bboxes = np.array(bboxes, dtype="float32")
    imagePaths = np.array(imagePaths)

    return images, labels, bboxes, imagePaths

def split_dataset(images, labels, bboxes, imagePaths, test_size=0.20):

    # partition the data into training and testing splits using 80% of
    # the data for training and the remaining 20% for testing
    split = train_test_split(images, labels, bboxes, imagePaths,
        test_size=test_size, random_state=42)
    # unpack the data split
    (trainImages, testImages) = split[:2] # images
    (trainLabels, testLabels) = split[2:4] # labels
    (trainBBoxes, testBBoxes) = split[4:6] # bounding boxes
    (trainPaths, testPaths) = split[6:] # image paths

    # convert NumPy arrays to PyTorch tensors
    (trainImages, testImages) = torch.tensor(trainImages),\
        torch.tensor(testImages)
    (trainLabels, testLabels) = torch.tensor(trainLabels),\
        torch.tensor(testLabels)
    (trainBBoxes, testBBoxes) = torch.tensor(trainBBoxes),\
        torch.tensor(testBBoxes)
    
    return trainImages, testImages, trainLabels, testLabels, trainBBoxes, testBBoxes

class CustomTensorDataset(Dataset):
    def __init__(self,tensors, transforms=None):
        self.tensors = tensors
        self.transforms = transforms

    def __len__(self):
        # return the size of the dataset
        return self.tensors[0].size(0)
    
    def __getitem__(self, index):
        # grab the image, label, and its bounding box coordinates
        image = self.tensors[0][index]
        label = self.tensors[1][index]
        bbox = self.tensors[2][index]
        # transpose the image such that its channel dimension becomes
        # the leading one
        image = image.permute(2, 0, 1)
        # check to see if we have any image transformations to apply
        # and if so, apply them
        if self.transforms:
            image = self.transforms(image)
            # TODO: what about the bounding box?
        # return a tuple of the images, labels, and bounding
        # box coordinates
        return (image, label, bbox)
    
###############################

def create_dataloaders(
        images_file_path: str,
        annotations_path: str,
        batch_size: int=8,
        pin_memory: bool=True,
        transforms=None,
        num_workers: int=NUM_WORKERS
):
    # Get the images, labels, bounding boxes, and image paths
    images, labels, bboxes, imagePaths = extractData(
        images_file_path=images_file_path,
        annotations_path=annotations_path
    )

    # Split the dataset into training and testing splits
    trainImages, testImages, trainLabels, testLabels, trainBBoxes, testBBoxes = split_dataset(
        images=images,
        labels=labels,
        bboxes=bboxes,
        imagePaths=imagePaths,
        test_size=0.20
    )

    if transforms is None:
        # define normalization transforms
        transforms = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            # T.Normalize(mean=MEAN, std=STD)
    ])

    # convert to PyTorch datasets
    trainDS = CustomTensorDataset(
        tensors=(trainImages, trainLabels, trainBBoxes),
        transforms=transforms)
    testDS = CustomTensorDataset(
        tensors=(testImages, testLabels, testBBoxes),
        transforms=transforms)
    
    print("[INFO] total training samples: {}...".format(len(trainDS)))
    print("[INFO] total test samples: {}...".format(len(testDS)))
    
    # calculate steps per epoch for training and validation set
    trainSteps = len(trainDS) // batch_size
    valSteps = len(testDS) // batch_size
    
    # create data loaders
    trainLoader = DataLoader(
        dataset=trainDS, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory) # enables fast data transfer to CUDA-enabled GPUs 
    testLoader = DataLoader(
        dataset=testDS, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_memory)
    
    return trainLoader, testLoader

# Test dataloader
def main():

    from pathlib import Path

    # Setup path to data folder
    data_path = Path("/home/kpatel2s/kpatel2s/object_detection/custom_object_detector/dataset/standford_dogs_mini_10")

    # image files paths and annotations
    images_file_path = os.path.join(data_path, "yolo", "image_paths.txt")
    annotations_path = os.path.join(data_path, "yolo", "annotations")
    
    # create dataloaders
    train_dataloader, test_dataloader = create_dataloaders(
        images_file_path=images_file_path,
        annotations_path=annotations_path,
        batch_size=8,
        pin_memory=True,
        transforms=None
    )

    # get one sample from the training dataloader
    images, labels, bboxes = train_dataloader.dataset[0]
    print(images.shape)


if __name__ == "__main__":
    main()
    








