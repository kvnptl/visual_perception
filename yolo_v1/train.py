import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
import os
from dataset import VOCDataset, Compose
from loss import YOLOLoss
from model import YOLOv1
from utils import intersection_over_union, non_max_suppression, mean_average_precision, cellboxes_to_boxes, get_bboxes, plot_image, save_checkpoint, load_checkpoint

# Seed
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

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

# Model
model = YOLOv1(split_size=7, num_boxes=2, num_classes=20)
model.to(device)

# Model summary
from torchinfo import summary
summary(YOLOv1(), (1, 3, 448, 448))

#  Optimizer and loss
optimizer = optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)
loss_fn = YOLOLoss(split_size=7, num_boxes=2, num_classes=20)

if LOAD_MODEL:
    load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

# Dataset and Dataloader
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

# Training
loop = tqdm(range(EPOCHS), leave=True)
mAP = 0.0

for idx, epoch in enumerate(loop):

    # for x, y in train_loader:
    #        x = x.to(device)
    #        for idx in range(8):
    #            bboxes = cellboxes_to_boxes(model(x))
    #            bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
    #            plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

    loss_val = train_fn(train_loader, model, optimizer, loss_fn)

    pred_boxes, target_boxes = get_bboxes(
        train_loader, model, iou_threshold=0.5, threshold=0.4
    )

    mean_avg_prec = mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
    )

    loop.set_postfix(loss=loss_val, mean_avg_prec=mean_avg_prec)

    if epoch % 10 == 0:
        if mean_avg_prec > mAP and mean_avg_prec > 0.9:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=f"best_model.pth")
            mAP = mean_avg_prec
