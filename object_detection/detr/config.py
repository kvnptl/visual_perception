from datetime import datetime
import os

SELF_DIR = os.path.dirname(__file__)
DATASET_DIR = "/home/kpatel2s/work/kpatel2s_datasets/pascal_voc_dataset"
PARENT_DIR = os.path.dirname(__file__)
DATASET_NAME = "pascal_voc"
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

LEARNING_RATE = 1e-4  # = 0.01
BATCH_SIZE = 8
NUM_EPOCHS = 5
NUM_WORKERS = os.cpu_count()

SEED = 8
PIN_MEMORY = True

LOAD_MODEL = False
LOAD_MODEL_FILE = "/home/kpatel2s/work/kpatel2s_datasets/carvana_dataset/my_checkpoint.pth.tar"

MODEL_LOG = f"unet_batch_{BATCH_SIZE}"
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")

# Verbose
PRINT_MODEL_SUMMARY = False
