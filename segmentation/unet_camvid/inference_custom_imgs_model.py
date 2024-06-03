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
import numpy as np
from PIL import Image
from line_profiler import LineProfiler

# Hyperparameters
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = config.IMAGE_HEIGHT
IMAGE_WIDTH = config.IMAGE_WIDTH
DATASET_DIR = config.DATASET_DIR
LOAD_MODEL_FILE = "/home/kpatel2s/work/visual_perception/segmentation/unet_camvid/results/CamVid/2024-05-31_04-10/model/best_model_100.pth"
SAVE_PREDICTIONS = True

# Directory with images
images_dir = "/home/kpatel2s/work/kpatel2s_datasets/CamVid/train"
groundtruth_dir = "/home/kpatel2s/work/kpatel2s_datasets/CamVid/train_labels"

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

def main(img_dir, gt_dir=None):

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

    image_files = sorted(os.listdir(img_dir))
    if gt_dir is not None:
        gt_files = sorted(os.listdir(gt_dir))

    # Output dir
    # Get the folder name from LOAD_MODEL_FILE
    target_dir = "/".join(LOAD_MODEL_FILE.split("/")[:-2])
    target_dir = os.path.join(target_dir, "custom_set_results")

    tqdm_loop = tqdm(image_files, total=len(image_files), desc="Inferece")

    # Process each image
    for idx, image_file in enumerate(tqdm_loop):
        image_path = os.path.join(images_dir, image_file)
        image = Image.open(image_path)
        if gt_dir is not None:
            gt_path = os.path.join(gt_dir, gt_files[idx])
            gt = Image.open(gt_path)
            gt = dataloader.adjust_mask(gt)
            transformed = test_transforms(image=np.array(image), mask=np.array(gt))
            image = transformed['image']
            gt = transformed['mask']
            gt = gt.unsqueeze(0).to(DEVICE)

            # Convert one-hot encoded mask to ckass indices
            if gt.shape[-1] == 3 or gt.shape[-1] == 32:
                gt = torch.argmax(gt, dim=-1)
        else:
            image = test_transforms(image=np.array(image))['image']
        image = image.unsqueeze(0).to(DEVICE)  # Add batch dimension and move to device

        # Test the unet_model
        with torch.inference_mode():
            with torch.no_grad():
                outputs = unet_model(image)
                preds = torch.argmax(outputs, dim=1)
                preds_cpu = preds.cpu()
                if SAVE_PREDICTIONS:
                    if gt_dir is not None:
                        utils.visualize(idx, image[0].permute(1, 2, 0).cpu(),
                                        preds_cpu[0], gt[0].cpu(), map_class_to_rgb=map_class_to_rgb, folder=target_dir)
                    else:
                        utils.visualize(idx, image[0].permute(1, 2, 0).cpu(),
                                    preds_cpu[0], map_class_to_rgb=map_class_to_rgb, folder=target_dir)

if __name__ == '__main__':
    main(img_dir=images_dir, gt_dir=groundtruth_dir)