# %% [markdown]
# # YOLOv1 architecture
# 
# This is a YOLO v1 architecture implementation using PyTorch.
# 
# References:https://youtu.be/n9_XyCGr-MI?si=GNjO1LAW429Ycsdh
# 

# %%
import torch
import torch.nn as nn

# %%
architecture_config = [
    (7, 64, 2, 3), # kernel, channels, stride, padding
    (2, 2), # kernel, stride

    (3, 192, 1, 1),
    (2, 2),

    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    (2, 2),
    # List: tuple(kernel, channels, stride, padding) and last one is the number of repeats
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], # 0
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    (2, 2),

    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

# %%
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             bias=False,
                             **kwargs)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

# %%
class YOLOv1(nn.Module):
    def __init__(self, in_channels: int=3, **kwargs):
        super().__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self.create_conv_layers(self.architecture)
        self.fcs = self.create_fcs(**kwargs)

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            # Conv block
            if type(x) == tuple and len(x) == 4:
                layers += [CNNBlock(in_channels=in_channels, 
                                  out_channels=x[1], 
                                  kernel_size=x[0], 
                                  stride=x[2], 
                                  padding=x[3])]
                in_channels = x[1]
                
            # Maxpool
            elif type(x) == tuple and len(x) == 2:
                layers += [nn.MaxPool2d(kernel_size=x[0], stride=x[1])]
            # Conv repeated block
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [CNNBlock(in_channels=in_channels, 
                                      out_channels=conv1[1], 
                                      kernel_size=conv1[0], 
                                      stride=conv1[2], 
                                      padding=conv1[3])]
                    layers += [CNNBlock(in_channels=conv1[1], 
                                      out_channels=conv2[1], 
                                      kernel_size=conv2[0], 
                                      stride=conv2[2], 
                                      padding=conv2[3])]
                    in_channels = conv2[1]

        return nn.Sequential(*layers) # *layers unpacks the list of layers
    
    def create_fcs(self, split_size=7, num_boxes=2, num_classes=20): # split_size is the grid size
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024 * S * S, out_features=4096),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=4096, out_features=S*S*(C + B*5)), # each cell is 30x1 (20 classes + (1+4) 1st box + (1+4) 2nd box), where (1+4): probability + x1, y1, x2, y2
        )

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(x)

# %%
from torchinfo import summary
summary(YOLOv1(), (1, 3, 448, 448))

# %% [markdown]
# # YOLO Loss 

# %% [markdown]
# <img src="figures/image.png" width=600>

# %%
from utils import intersection_over_union

class YOLOLoss(nn.Module):
    def __init__(self, split_size=7, num_boxes=2, num_classes=20):
        super().__init__()
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        self.mse = nn.MSELoss(reduction="sum") # sum of all elements
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # We have to reshape as YOLO output is (1, 1470), so convert it to (1, 7, 7, 30)
        predictions = predictions.reshape(-1, 
                                         self.split_size, 
                                         self.split_size, 
                                         self.num_classes + self.num_boxes * 5) # -1 means don't touch the first dimension, it could be anything (here batch size)

        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25]) # [...]: means take all elements, e.g. from (1, 7, 7, 30), take (1, 7, 7) as it is
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3).float() # Identity matrix

        # Bounding box regression
        box_predictions = exists_box * (
            best_box * predictions[..., 26:30] + (1 - best_box) * predictions[..., 21:25]
        )
        box_targets = exists_box * target[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * \
                                    torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6)) # sign is used to preserve the sign, as we are removing while calculating square root

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )


        # Object loss
        pred_box = (
            best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21]
        )

        # (N, S, S) -> (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # No object loss
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1), 
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1), 
        )

        # Class loss
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2), # end_dim=-2: flatten the last dim (N, S, S, 20) -> (N*S*S, 20)
            torch.flatten(exists_box * target[..., :20], end_dim=-2),
        )

        # Total loss
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss

# %% [markdown]
# # Dataset preparation

# %%
import pandas as pd
import os
from PIL import Image

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

            i, j = int(self.split_size * x), int(self.split_size * y)
            x_cell, y_cell = self.split_size * x - i, self.split_size * y - j

            width_cell, height_cell = (
                width * self.split_size,
                height * self.split_size,
            )

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix

# %% [markdown]
# # Training

# %%
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from utils import intersection_over_union, non_max_suppression, mean_average_precision, cellboxes_to_boxes, get_bboxes, plot_image, save_checkpoint, load_checkpoint

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Hyperparameters
learning_rate = 2e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
weight_decay = 0
EPOCHS = 100
NUM_WORKERS = os.cpu_count()
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "/srv/disk1/datasets/kpatel2s_datasets/pascal_voc_dataset/images"
LABEL_DIR = "/srv/disk1/datasets/kpatel2s_datasets/pascal_voc_dataset/labels"

# %%
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

# %%
def train_fn(train_loader, model, optimizer, loss_fn):
    # loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        # loop.set_postfix(loss=loss.item())

    # print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")
    return sum(mean_loss)/len(mean_loss)

# %%
model = YOLOv1(split_size=7, num_boxes=2, num_classes=20)
model.to(device)
optimizer = optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)
loss_fn = YOLOLoss(split_size=7, num_boxes=2, num_classes=20)

# %%
if LOAD_MODEL:
    load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

train_dataset = VOCDataset(
    csv_file="/srv/disk1/datasets/kpatel2s_datasets/pascal_voc_dataset/100examples.csv",
    transform=transform,
    img_dir=IMG_DIR,
    label_dir=LABEL_DIR,
)

test_dataset = VOCDataset(
    csv_file="/srv/disk1/datasets/kpatel2s_datasets/pascal_voc_dataset/100examples.csv",
    transform=transform,
    img_dir=IMG_DIR,
    label_dir=LABEL_DIR,
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    shuffle=True,
    drop_last=False,
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    shuffle=False,
    drop_last=False,
)

# %%
from utils import get_bboxes

loop = tqdm(range(EPOCHS), leave=True)

for idx, epoch in enumerate(loop):

    # for x, y in train_loader:
    #        x = x.to(device)
    #        for idx in range(8):
    #            bboxes = cellboxes_to_boxes(model(x))
    #            bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
    #            plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

    pred_boxes, target_boxes = get_bboxes(
        train_loader, model, iou_threshold=0.5, threshold=0.4
    )

    mean_avg_prec = mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
    )
    
    # loop.set_postfix(mean_avg_prec=mean_avg_prec)
    # print(f"Mean Average Precision: {mean_avg_prec}")

    # if mean_avg_prec > 0.9:
    #     checkpoint = {
    #         "state_dict": model.state_dict(),
    #         "optimizer": optimizer.state_dict(),
    #     }
    #     save_checkpoint(checkpoint, "overfit.pth.tar")
    #     import time
    #     time.sleep(10)

    loss_val = train_fn(train_loader, model, optimizer, loss_fn)

    loop.set_postfix(loss=loss_val, mean_avg_prec=mean_avg_prec)

# %%



