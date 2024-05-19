## UNet based semantic segmentation on CamVid dataset

The project is structured to facilitate training a U-Net model for segmenting driving scenes from CamVid dataset, providing both standalone Python scripts and a Jupyter notebook for interactive experimentation. This setup is ideal for those interested in applying deep learning to real-world vision tasks in autonomous driving and related fields.

## Dataset

- CamVid (Cambridge-Driving Labeled Video Database): <https://www.kaggle.com/datasets/carlolepelaars/camvid>
- The dataset is split up as follows:
  - 367 training pairs
  - 101 validation pairs
  - 233 test pairs
- 32 classes

## Functionalities

- Store best model based on val loss
- Plot loss and accuracy curves
- Config file to store hyperparameters
- Produce detailed model summary
- Modularize code

## How to train U-Net on CamVid dataset

- Run train script:

  ```bash
  python train.py
  ```

  - Check results in `results` folder, it should have model checkpoints, loss and accuracy curves, and sample val predictions

## TODO

- [x] Model monitoring code
  - [x] Loss curves, Accuracy curves
- [ ] (if possible) Integrate focal loss to mitigate class imbalance
- [ ] Add regularization to prevent overfitting, i.e. dropout
- [ ] Add learning rate scheduler
- [ ] Add early stopping
- [ ] Fix Tensorboard logging
- [ ] Add IoU loss for semantic segmentation
