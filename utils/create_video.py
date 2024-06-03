import cv2
import glob
import re
import os
from tqdm import tqdm

# Input image directory
IMG_PATH = '/images/*.png'
# Output video path
VIDEO_PATH = '/dir/video/video.mp4'
# FPS of the output video
FPS = 2

if not os.path.exists(os.path.dirname(VIDEO_PATH)):
    os.makedirs(os.path.dirname(VIDEO_PATH))

img_array = []
numbers = re.compile(r'(\d+)') # regex to sort the images

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(numbers, text)]

print("Step 1/3: Extracting images from folder")
for filename in tqdm(glob.glob(IMG_PATH)):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append((img, filename))

img_array.sort(key=lambda img: natural_keys(img[1]))

out = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), FPS, size)
print(f"\nStep 2/3: Video creation started. Saving to {VIDEO_PATH}")

for img, filename in tqdm(img_array):
    out.write(img)
out.release()

print(f"\nStep 3/3: DONE! Video saved to {VIDEO_PATH}")