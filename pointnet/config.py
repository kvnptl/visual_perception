import os
from datetime import datetime

DATASET_NAME = "ModelNet10"
PARENT_DIR = os.path.dirname(__file__)
DATASET = os.path.join(PARENT_DIR, "dataset")
NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 128
PIN_MEMORY = True
SEED = 8
NUM_CLASSES = 10

INPUT_POINT_CLOUD_SIZE = 1024

LR_RATE = 1e-3
EPOCHS = 15

MODEL_LOG = "pointnet_batch_32"

LOAD_MODEL = False
MODEL_PATH = "model_50.pth"

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")