"""
This script is used to concatenate the images in the results folder.

3 images are concatenated:
    1. The original image
    2. Ground truth image
    3. Predicted image

"""

import os
import cv2
import numpy as np

# 1. Get the paths to the images
orig_dir = '/original_images/dir'
gt_dir = '/ground_truth/dir'
pred_dir = '/predictions/dir'

concat_dir = '/path/to/store/concat'

# 2. Get the image names
orig_images = os.listdir(orig_dir)
# from all the images, get only the images that have the word 'orig' in them
orig_images = [image for image in orig_images if 'orig' in image]

gt_images = os.listdir(gt_dir)
# from all the images, get only the images that have the word 'gt' in them
gt_images = [image for image in gt_images if 'gt' in image]

pred_images = os.listdir(pred_dir)
# from all the images, get only the images that have the word 'pred' in them
pred_images = [image for image in pred_images if 'pred' in image]

# 3. Sort the images
orig_images.sort()
gt_images.sort()
pred_images.sort()

# 4. Create a directory to store the concatenated images
if not os.path.exists(concat_dir):
    os.mkdir(concat_dir)

# 5. Concatenate the images
for orig_image, gt_image, pred_image in zip(orig_images, gt_images, pred_images):
    # 5.1. Get the images
    orig_img = cv2.imread(os.path.join(orig_dir, orig_image))
    gt_img = cv2.imread(os.path.join(gt_dir, gt_image))
    pred_img = cv2.imread(os.path.join(pred_dir, pred_image))

    # put tags on the images, left to right: original, ground truth, predicted
    # put text on top-left corner, in blue color
    cv2.putText(orig_img, 'OG', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(gt_img, 'GT', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(pred_img, 'Pred', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # 5.2. Concatenate the images
    concat_img = np.concatenate((orig_img, gt_img, pred_img), axis=1)

    final_image_name = orig_image.split('_')[0] + '_orig_gt_pred.png'

    # 5.3. Save the concatenated image
    cv2.imwrite(os.path.join(concat_dir, final_image_name), concat_img)

# 6. Print some statistics
print('Number of images: ', len(orig_images))