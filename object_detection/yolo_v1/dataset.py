import torch
import pandas as pd
import os
from PIL import Image

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, split_size=7, num_boxes=2, num_classes=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes) # converting to tensor so that we can apply transforms

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.split_size, self.split_size, self.num_classes + self.num_boxes * 5))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            """
            # NOTE: For the better understanding of the below code, 
            # refer to the video: https://youtu.be/zgbPj4lSc58?si=n1Vaemxrt0GS5O0O&t=260
            """

            # i, j is the index of the grid cell, it assigns which grid cell the box belongs to
            i, j = int(self.split_size * x), int(self.split_size * y)
            # x_cell, y_cell is the offset of the box in the grid cell
            # Basically, in the grid cell (i, j), how much the box is offset from the left-top corner of the grid cell
            x_cell, y_cell = self.split_size * x - i, self.split_size * y - j

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.split_size,
                height * self.split_size,
            )

            # Check the 20th element in the vector at (i, j), remember that the it's SxSx(C+(B*5)) matrix
            # Assuming we have 20 classes, so the initial 0 to 19th elements are for classes
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1 # Objectness score
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1 # Class score, following one-hot encoding

        return image, label_matrix