import torch
import torchvision
from dataloader import CarvanaDataset
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import os


def save_checkpoint(state, filename="mt_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    dataset_dir,
    dataset_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):

    # Get total number of samples
    num_samples = len(os.listdir(dataset_dir))

    val_size = int(0.1 * num_samples)
    train_size = num_samples - val_size

    # Split the dataset
    train_indices, val_indices = random_split(
        list(range(num_samples)), [train_size, val_size])

    train_dataset = CarvanaDataset(
        img_dir=dataset_dir,
        mask_dir=dataset_maskdir,
        transform=train_transform,
        indices=train_indices
    )

    val_dataset = CarvanaDataset(
        img_dir=dataset_dir,
        mask_dir=dataset_maskdir,
        transform=val_transform,
        indices=val_indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    # TODO: verify this dataset split and dataloader function

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            # number of elements in the tensor
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}")

    # Divide by len(loader) because to calculate mean of dice score
    print(f"Dice score: {dice_score/len(loader)}")

    model.train()


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):

    if not os.path.exists(folder):
        os.makedirs(folder)

    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()
