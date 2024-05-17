import torch
import torch.nn as nn
import torchvision
import torchmetrics
import model
import config
import utils
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import os

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_WORKERS = config.NUM_WORKERS
IMAGE_HEIGHT = config.IMAGE_HEIGHT
IMAGE_WIDTH = config.IMAGE_WIDTH
PIN_MEMORY = config.PIN_MEMORY
DATASET_DIR = config.DATASET_DIR
LOAD_MODEL_FILE = "/home/kpatel2s/work/visual_perception/segmentation/unet_camvid/results/CamVid/2024-05-18_00-09/model/best_model_10.pth"

# Dataset prep
# Get the class label csv file
classes = pd.read_csv(DATASET_DIR + '/class_dict.csv', index_col=0)

# There are total 32 classes
n_classes = len(classes)
print(f'[INFO] Total number of classes: {n_classes}')

model = model.UNet(in_channels=3, out_channels=n_classes).to(DEVICE)

# Load the model
model.load_state_dict(torch.load(LOAD_MODEL_FILE))
print(f"[INFO] Model loaded from {LOAD_MODEL_FILE}")

# Test
test_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

test_dataloader = utils.get_loaders(
    dataset_dir=DATASET_DIR,
    batch_size=BATCH_SIZE,
    transform=test_transforms,
    set_type="test",
    pin_memory=PIN_MEMORY,
    num_workers=NUM_WORKERS
)


loss_fn = nn.CrossEntropyLoss()  # for multi class, use cross entropy

# Output dir
# Get the folder name from LOAD_MODEL_FILE
target_dir = "/".join(LOAD_MODEL_FILE.split("/")[:-2])
target_dir = os.path.join(target_dir, "testset_results")

# Test the model
test_loss, test_acc, test_dice_score, test_iou_score = 0.0, 0.0, 0.0, 0.0

tqdm_loop = tqdm(test_dataloader, desc="Test")

with torch.inference_mode():
    for idx, (data, target) in enumerate(tqdm_loop):
        data = data.to(DEVICE)
        target = target.to(DEVICE)

        with torch.no_grad():
            outputs = model(data)
            loss = loss_fn(outputs, target)

            preds = torch.argmax(outputs, dim=1)
            x_cpu = data.cpu()
            preds_cpu = preds.cpu()
            y_cpu = target.cpu()
            utils.visualize(idx, x_cpu[0].permute(1, 2, 0),
                            preds_cpu[0], y_cpu[0], folder=target_dir)

        test_loss += loss.item()
        test_acc += utils.pixel_accuracy(outputs, target)
        test_dice_score += utils.dice_score_fn(outputs, target)
        test_iou_score += utils.iou_score_fn(outputs, target)

        tqdm_loop.set_postfix(
            loss=test_loss / (idx + 1),
            acc=test_acc / (idx + 1),
            dice_score=test_dice_score / (idx + 1),
            iou_score=test_iou_score / (idx + 1)
        )

test_loss = test_loss / len(test_dataloader)
test_acc = test_acc / len(test_dataloader)
test_dice_score = test_dice_score / len(test_dataloader)
test_iou_score = test_iou_score / len(test_dataloader)

print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Dice Score: {test_dice_score:.4f}, Test IoU Score: {test_iou_score:.4f}")
