from datetime import datetime
import os

SELF_DIR = os.path.dirname(__file__)
DATASET_DIR = "/home/kpatel2s/work/kpatel2s_datasets"
DATASET_NAME = "pizza_steak_sushi"
IMG_SIZE = 224
NUM_CLASSES = 3

LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 3
NUM_WORKERS = os.cpu_count()

PATCH_SIZE = 16
INPUT_CHANNELS = 3
EMBEDDING_DIM = 768
NUM_HEADS = 12
NUM_TRANSFORMER_LAYERS = 12
MLP_SIZE = 3072

SEED = 8
PIN_MEMORY = True

LOAD_MODEL = False
LOAD_MODEL_FILE = "/home/kpatel2s/work/kpatel2s_datasets/carvana_dataset/my_checkpoint.pth.tar"

MODEL_LOG = f"unet_batch_{BATCH_SIZE}"
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")

# Verbose
PRINT_MODEL_SUMMARY = True
