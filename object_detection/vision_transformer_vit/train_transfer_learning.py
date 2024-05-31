import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
from torchinfo import summary
import data_setup, engine, utils, config, model
from pathlib import Path
from PIL import Image
import random

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
DATASET_DIR = config.DATASET_DIR
DATASET_NAME = config.DATASET_NAME
IMG_SIZE = config.IMG_SIZE
NUM_CLASSES = config.NUM_CLASSES
NUM_EPOCHS = config.NUM_EPOCHS
LEARNING_RATE = config.LEARNING_RATE
NUM_WORKERS = config.NUM_WORKERS
BATCH_SIZE = config.BATCH_SIZE
PATCH_SIZE = config.PATCH_SIZE
INPUT_CHANNELS = config.INPUT_CHANNELS
EMBEDDING_DIM = config.EMBEDDING_DIM
NUM_HEADS = config.NUM_HEADS
NUM_TRANSFORMER_LAYERS = config.NUM_TRANSFORMER_LAYERS
MLP_SIZE = config.MLP_SIZE
PRINT_MODEL_SUMMARY = config.PRINT_MODEL_SUMMARY

# Download dataset
utils.download_dataset()

# Prepare dataset
data_path = Path(DATASET_DIR)
image_path = data_path / DATASET_NAME

# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

# Create ViT model
model_weights = torchvision.models.ViT_B_16_Weights.DEFAULT # B_16 means Base model with 16 patch size
model = torchvision.models.vit_b_16(weights=model_weights).to(device)

# Freeze all base model layers
for param in model.encoder.parameters():
    param.requires_grad = False
for param in model.conv_proj.parameters():
    param.requires_grad = False

# Update classifier head
model.heads = nn.Sequential(
    nn.Linear(in_features=EMBEDDING_DIM, out_features=NUM_CLASSES)
)

if PRINT_MODEL_SUMMARY:
    summary(model=model,
            input_size=(1, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            row_settings=["var_names"])

# Data transforms
vit_transforms = model_weights.transforms() # take the transforms from the original model

# Setup Dataloaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=vit_transforms,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS)


# Loss and Optimizer
loss_fn = nn.CrossEntropyLoss() # the paper didn't mention a loss function
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=LEARNING_RATE,
                             weight_decay=0.1)


# Set seeds
utils.set_seeds()

# Train the model
results = engine.train(model=model,
                      train_dataloader=train_dataloader,
                      test_dataloader=test_dataloader,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      epochs=NUM_EPOCHS,
                      device=device)

# Plot loss and accuracy curves
utils.plot_loss_curves(results=results, save_fig=True)

# Save the model
save_filename = "vit_b_16_10_epochs.pth"
utils.save_model(model=model,
                 target_dir="models",
                 model_name=save_filename)

# Predict and plot an image
def pred_and_plot_img(model,
                      img_path,
                      class_names,
                      img_size,
                      transform,
                      device):
    img = Image.open(img_path)

    if transform is not None:
        img_transform = transform
    else:
        img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    model.to(device)

    model.eval()
    
    with torch.inference_mode():
        img_tensor = img_transform(img).unsqueeze(0)
        img_tensor = img_tensor.to(device)
        prediction = model(img_tensor)

    pred_probs = torch.softmax(prediction, dim=1)
    pred_class = torch.argmax(pred_probs, dim=1)

    plt.figure()
    plt.title(f"Prediction: {class_names[pred_class]} | Prob: {pred_probs.max():.3f}")
    plt.imshow(img)
    # Save the predicted image
    target_dir = os.path.join(
            config.SELF_DIR, "results", config.DATASET_NAME, config.TIMESTAMP, "pred")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    file_name = os.path.basename(img_path).split(".")[0] + "_pred.jpg"
    plt.savefig(os.path.join(target_dir, file_name))


num_imgs = 3
test_img_path_list = list(Path(test_dir).glob("*/*.jpg"))
test_img_path_sample = random.sample(population=test_img_path_list, k=num_imgs)

for test_img in test_img_path_sample:
    pred_and_plot_img(model=model,
                    img_path=test_img,
                    class_names=class_names,
                    img_size=(224, 224),
                    transform=None,
                    device=device)