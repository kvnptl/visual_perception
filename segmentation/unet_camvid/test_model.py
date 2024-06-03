import torch
import torch.nn as nn
import torchvision
import torchmetrics
import model
import config
import utils
import dataloader
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import os
from line_profiler import LineProfiler

# Hyperparameters
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_WORKERS = config.NUM_WORKERS
IMAGE_HEIGHT = config.IMAGE_HEIGHT
IMAGE_WIDTH = config.IMAGE_WIDTH
PIN_MEMORY = config.PIN_MEMORY
DATASET_DIR = config.DATASET_DIR
LOAD_MODEL_FILE = "/home/kpatel2s/work/visual_perception/segmentation/unet_camvid/results/CamVid/2024-05-31_04-10/model/best_model_100.pth"
SAVE_PREDICTIONS = True

# Dataset prep
# Get the class label csv file
classes = pd.read_csv(DATASET_DIR + '/class_dict.csv', index_col=0)

cls2rgb = {cl:list(classes.loc[cl, :]) for cl in classes.index}
map_class_to_rgb = {i: rgb for i, rgb in enumerate(cls2rgb.values())} # a new dictionary with integer keys

# There are total 32 classes
n_classes = len(classes)
print(f'[INFO] Total number of classes: {n_classes}')

def profile(func):
    def wrapper(*args, **kwargs):
        lp = LineProfiler()
        lp.add_function(func)  # Add the function you want to profile
        result = lp(func)(*args, **kwargs)
        lp.print_stats(output_unit=1e-3)
        return result
    return wrapper

def main():

    # Set seed
    utils.set_seed(seed=config.SEED)

    unet_model = model.UNet(in_channels=3, out_channels=n_classes).to(DEVICE)

    # Load the unet_model
    unet_model.load_state_dict(torch.load(LOAD_MODEL_FILE))
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

    test_dataloader = dataloader.get_loaders(
        dataset_dir=DATASET_DIR,
        batch_size=BATCH_SIZE,
        transform=test_transforms,
        set_type="train",
        pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS
    )

    # Output dir
    # Get the folder name from LOAD_MODEL_FILE
    target_dir = "/".join(LOAD_MODEL_FILE.split("/")[:-2])
    target_dir = os.path.join(target_dir, "train_set_results")

    # Test the unet_model
    test_loss, test_acc, test_dice_score, test_iou_score = 0.0, 0.0, 0.0, 0.0

    tqdm_loop = tqdm(test_dataloader, desc="Test")

    with torch.inference_mode():
        for idx, (data, target) in enumerate(tqdm_loop):
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            with torch.no_grad():
                outputs = unet_model(data)

                preds = torch.argmax(outputs, dim=1)
                x_cpu = data.cpu()
                preds_cpu = preds.cpu()
                y_cpu = target.cpu()
                if SAVE_PREDICTIONS:
                    utils.visualize(idx, x_cpu[0].permute(1, 2, 0),
                                    preds_cpu[0], y_cpu[0], map_class_to_rgb, folder=target_dir)

            test_acc += utils.pixel_accuracy(outputs, target)
            test_dice_score += utils.dice_score_fn(outputs, target)
            test_iou_score += utils.iou_score_fn(outputs, target)

            tqdm_loop.set_postfix(
                acc=test_acc / (idx + 1),
                dice_score=test_dice_score / (idx + 1),
                iou_score=test_iou_score / (idx + 1)
            )

    test_acc = test_acc / len(test_dataloader)
    test_dice_score = test_dice_score / len(test_dataloader)
    test_iou_score = test_iou_score / len(test_dataloader)

    print(f"Test Acc: {test_acc*100:.2f}, Test Dice Score: {test_dice_score*100:.2f}, Test IoU Score: {test_iou_score*100:.2f}")

if __name__ == '__main__':
    main()