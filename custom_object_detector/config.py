import os

DATASET = "dataset/standford_dogs_mini_10"
NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 32
PIN_MEMORY = True
SEED = 8
IMAGE_SIZE = 224
NUM_CLASSES = 10

LR_RATE = 1e-4
EPOCHS = 10

LOAD_MODEL = False
MODEL_PATH = "model.pt"

# ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

CLASS_NAMES = ['Chihuahua', 
           'Golden_retriever', 
           'Welsh_springer_spaniel', 
           'German_shepherd', 
           'Doberman', 
           'Boxer', 
           'Siberian_husky', 
           'Pug', 
           'Pomeranian', 
           'Cardigan']