from datetime import datetime

DATASET_DIR = "/home/kpatel2s/work/kpatel2s_datasets/CamVid"
PARENT_DIR = "/home/kpatel2s/work/visual_perception/segmentation/unet_camvid"
DATASET_NAME = "CamVid"
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_WORKERS = 4

SEED = 8
PIN_MEMORY = True

LOAD_MODEL = False
LOAD_MODEL_FILE = "/home/kpatel2s/work/kpatel2s_datasets/carvana_dataset/my_checkpoint.pth.tar"

MODEL_LOG = f"unet_batch_{BATCH_SIZE}"
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")

# Verbose
PRINT_MODEL_SUMMARY = False
