### Steps to run YOLOv1 ###

Reference: [YOLOv1 from Scratch](https://youtu.be/n9_XyCGr-MI?si=t9uYmwKx2zi1blxC)

#### Note:
- make sure to update the dataset path

### Steps
- The code is modular,
    - `train.py`: Training script
    - `dataset.py`: dataloader and preprocessing 
    - `model.py`: YOLOv1 model
    - `loss.py`: YOLOv1 loss
    - `utils.py`: code for iou, nms, mAP, etc.
- Run `train.py` to train the model (tune the hyperparameters)

- The same code also available as a single script, `full_code.py`

### TODO:
- Model tracking is not available, loss vs acc graphs, etc.
- Improve `inference.py` script for better visualization
