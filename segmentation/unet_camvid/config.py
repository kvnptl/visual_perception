from datetime import datetime
import os

DATASET_DIR = "/home/kpatel2s/work/kpatel2s_datasets/CamVid"
PARENT_DIR = os.path.dirname(__file__)
DATASET_NAME = "CamVid"
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

LEARNING_RATE = 1e-4  # = 0.01
BATCH_SIZE = 8
NUM_EPOCHS = 100
NUM_WORKERS = 8

SEED = 8
PIN_MEMORY = True

LOAD_MODEL = False
LOAD_MODEL_FILE = "/home/kpatel2s/work/kpatel2s_datasets/carvana_dataset/my_checkpoint.pth.tar"

MODEL_LOG = f"unet_batch_{BATCH_SIZE}"
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")

# Verbose
PRINT_MODEL_SUMMARY = True
