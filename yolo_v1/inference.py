import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim
from model import YOLOv1
from dataset import VOCDataset, Compose
from utils import get_bboxes, load_checkpoint


LOAD_MODEL_FILE = "/home/kpatel2s/kpatel2s/best_model.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

IMG_DIR = "/srv/disk1/datasets/kpatel2s_datasets/pascal_voc_dataset/images"
LABEL_DIR = "/srv/disk1/datasets/kpatel2s_datasets/pascal_voc_dataset/labels"

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

model = YOLOv1(split_size=7, num_boxes=2, num_classes=20)
model.to(device)

opimizer = optim.Adam(
    model.parameters(), lr=2e-5, weight_decay=0
)

load_checkpoint(checkpoint=torch.load(LOAD_MODEL_FILE), model=model, optimizer=opimizer)

test_dataset = VOCDataset(
    csv_file="/srv/disk1/datasets/kpatel2s_datasets/pascal_voc_dataset/100examples.csv",
    transform=transform,
    img_dir=IMG_DIR,
    label_dir=LABEL_DIR
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False,
)

pred_boxes, true_boxes = get_bboxes(
    test_loader, model, iou_threshold=0.5, threshold=0.4
)

from utils import plot_image
plot_image(test_dataset[0][0].permute(1,2,0).to("cpu"), pred_boxes)
plot_image(test_dataset[0][0].permute(1,2,0).to("cpu"), true_boxes)

from utils import mean_average_precision

mean_avg_prec = mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint"
)

print(mean_avg_prec)



