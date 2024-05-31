
### 2.1 Create Dataset and Dataloaders (script mode)

"""
Contains functionality for creating PyTorch DataLoader's for 
image classification data.
"""

import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int = 16,
        num_workers: int = NUM_WORKERS
):
        # Follow this style guide from Google style guide for Python
        """Create training and testing DataLoaders.

        Takes in a training directory and a testing directory path and turns them into
        PyTorch Dataset and then into PyTorch Dataloaders.

        Args:
            train_dir (str): Path to training directory.
            test_dir (str): Path to testing directory.
            transform (callable): Optional transform to be applied on a sample.
            batch_size (int, optional): Number of samples per batch. Defaults to 16.
            num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to os.cpu_count().

        Returns:
            A tuple of (train_dataloader, test_dataloader, class_names)
        
        Example usage:
            train_dataloader, test_dataloader, class_names = create_dataloaders(
                train_dir=path_to_train_dir,
                test_dir=path_to_test_dir,
                transform=transform,
                batch_size=32,
                num_workers=4
            )
        """

        train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                          transform=transform, # transforms to perform on data (images)
                                          target_transform=None) # transforms to perform on labels (if necessary)

        test_data = datasets.ImageFolder(root=test_dir,
                                        transform=transform)

        class_names = train_data.classes

        train_dataloader = DataLoader(dataset=train_data,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=True) # enables fast data transfer to CUDA-enabled GPUs 

        test_dataloader = DataLoader(dataset=test_data,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     pin_memory=True)

        return train_dataloader, test_dataloader, class_names

# test dataloader
def main():

    import torchvision
    import os

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # defaults to imagenet IMAGENET1K_V1
    auto_transform = weights.transforms() # returns a transform.Compose object

    from pathlib import Path

    # Setup path to data folder
    data_path = Path("/home/kpatel2s/kpatel2s/pytorch-practice/data/")
    image_path = data_path / "pizza_steak_sushi"

    # Setup train and testing paths
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    train_dir, test_dir

    NUM_WORKERS = os.cpu_count()
    BATCH_SIZE = 32

    train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_dir,
                                                                        test_dir=test_dir,
                                                                        transform=auto_transform,
                                                                        batch_size=BATCH_SIZE,
                                                                        num_workers=NUM_WORKERS)

    print(f"train_dataloader: {train_dataloader}")
    print(f"test_dataloader: {test_dataloader}")
    print(f"class_names: {class_names}")
        
if __name__ == "__main__":
    main()