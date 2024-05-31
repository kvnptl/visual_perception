from pathlib import Path
import torchvision
from torch import nn
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
import config
import os

# Setup path to data folder
data_path = Path(config.DATASET_DIR)
image_path = data_path / config.DATASET_NAME

test_dir = image_path / "test"

def evaluate_model(model,
                   dataloader,
                   device):
    
    model.eval()

    test_accuracy = 0

    for batch, (X, y) in enumerate(dataloader):
        with torch.inference_mode():
            prediction = model(X.to(device)).to(device)
            test_accuracy += ((prediction.argmax(dim=1) == y.to(device)).sum().item())

    print(f"\nTest accuracy: {test_accuracy/len(dataloader) * 100:.2f} %\n")

# Create simple transform
data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

test_data = datasets.ImageFolder(root=test_dir, 
                                 transform=data_transform)

test_dataloader = torch.utils.data.DataLoader(test_data,
                                             batch_size=1,
                                             shuffle=False)

# Get class names as a list
class_names = test_data.classes

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load the model architecture
model = torchvision.models.vit_b_16().to(device)
# Update the classifier head 
model.heads = nn.Sequential(
    nn.Linear(in_features=config.EMBEDDING_DIM, out_features=config.NUM_CLASSES)
).to(device)

# load the model weights
weights_file = os.path.join(config.SELF_DIR, "models", "vit_b_16_10_epochs.pth")
model.load_state_dict(torch.load(weights_file))

evaluate_model(model=model, dataloader=test_dataloader, device=device)