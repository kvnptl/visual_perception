from pathlib import Path
import os
import pprint
import matplotlib.pyplot as plt
import random

import torch
import torchvision
import torchmetrics

import model as m
import dataset
import config
from engine import test_step
from utils import crawl_through_dir, pred_and_plot_img

num_classes = config.NUM_CLASSES

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def inference_custom_dataset(
        model: torch.nn.Module,
        image_dir: str):
    
    model.to(device)

    num_imgs = 9
    test_img_path_list = crawl_through_dir(image_dir)
    test_img_path_sample = random.sample(test_img_path_list, num_imgs)

    # Set figure size
    plt.figure(figsize=(20, 20)) 

    # Set subplot parameters
    plt.subplots(nrows=3, ncols=3, figsize=(12, 12))

    for i, test_img in enumerate(test_img_path_sample):
        # plot output images
        plt.subplot(3, 3, i+1)
        image, label, gt_label = pred_and_plot_img(
            model=model,
            img_path=test_img,
            class_names=config.CLASS_NAMES,
            img_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
            transform=None,
            device=device)
        plt.axis("off")
        plt.imshow(image)
        if gt_label is None:
            plt.title(f"Pred: {label.lower()}")
        else:
            plt.title(f"Pred: {label.lower()} | GT: {gt_label.lower()}")

    plt.tight_layout(pad=2.0)
    plt.savefig("custom_inference_output.jpg")

def inference_test_dataset(
    model: torch.nn.Module,
    loss_fn: tuple,
    device: torch.device,
    images_path: str,
    annotations_path: str,
    pin_memory: bool=True,
    num_workers: int=1
):

    test_dataloader = dataset.create_dataloader(
        images_path=images_path,
        annotations_path=annotations_path,
        subset="test",
        batch_size=1,
        pin_memory=pin_memory,
        transforms=None,
        num_workers=num_workers
    )

    model.to(device)
    
    test_loss, test_acc, test_cls_loss, test_cls_acc, test_bbox_loss, test_bbox_acc = test_step(
        model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        device=device,
    )

    # Print results as a dictionary
    results_dict = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_cls_loss": test_cls_loss,
        "test_cls_acc": test_cls_acc,
        "test_bbox_loss": test_bbox_loss,
        "test_bbox_acc": test_bbox_acc
    }

    pprint.pprint(results_dict)


def model_config(model_path):
    # Create the network
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    basemodel = torchvision.models.resnet50(weights=weights)

    model = m.ObjectDetector(basemodel, num_classes)

    print("[INFO] Loading model...")
    # load the model weights
    
    # check if file exists
    if not os.path.exists(model_path):
        # File does not exist
        weights_file = os.path.join(config.PARENT_DIR, "models", model_path)

    model.load_state_dict(torch.load(weights_file))

    return model    

def main():
    inference_on_custom_imgs = False

    model = model_config(config.MODEL_PATH)

    if inference_on_custom_imgs:
        image_dir = "/home/kpatel2s/kpatel2s/object_detection/custom_object_detector/dataset/standford_dogs_mini_10/images"

        inference_custom_dataset(
            model=model,
            image_dir=image_dir
        )

    else:
        # On test dataset
        
        # Setup path to data folder
        data_path = Path(config.DATASET)

        # image files paths and annotations
        images_path = os.path.join(data_path, "images")
        annotations_path = os.path.join(data_path, "yolo", "annotations")

        NUM_WORKERS = config.NUM_WORKERS
        PIN_MEMORY = config.PIN_MEMORY

        # Loss and optimizer
        # Two loss functions
        classLoss_function = torch.nn.CrossEntropyLoss()
        bboxLoss_function = torch.nn.MSELoss()
        loss_fn = (classLoss_function, bboxLoss_function)

        # Inference on test dataset
        inference_test_dataset(
            model=model,
            loss_fn=loss_fn,
            device=device,
            images_path=images_path,
            annotations_path=annotations_path,
            pin_memory=PIN_MEMORY,
            num_workers=NUM_WORKERS
        )

    print("Done!")


if __name__ == "__main__":
    main()