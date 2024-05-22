## UNet based semantic segmentation on CamVid dataset

The project is structured to facilitate training a U-Net model for segmenting driving scenes from CamVid dataset, providing both standalone Python scripts and a Jupyter notebook for interactive experimentation. This setup is ideal for those interested in applying deep learning to real-world vision tasks in autonomous driving and related fields.

- Visualize the UNet model architecture graph here: [Network Graph (SVG)](media/unet_model_graph.gv.svg)

## Results

<p float="left">
  <img src="media/_image_0.png"/>
  <br>
  <img src="media/_image_1.png"/>
  <br>
  <img src="media/_image_2.png"/>
  <br>
  <img src="media/_image_3.png"/>
</p>

## Loss and accuracy curves

<p float="left">
  <img src="media/loss_and_acc_curves.png"/>
</p>

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
  python train.py <GPUDeviceIndex>
  ```

  - Check results in `results` folder, it should have model checkpoints, loss and accuracy curves, and sample val predictions

## TODO

- [x] Model monitoring code
  - [x] Loss curves, Accuracy curves
- [x] Train with bigger image size, e.g. 512x512
- [x] Add regularization to prevent overfitting, i.e. dropout
- [ ] (if possible) Integrate focal loss to mitigate class imbalance
- [ ] Add IoU loss for semantic segmentation
- [ ] Add learning rate scheduler
- [ ] Add early stopping
- [ ] Fix Tensorboard logging
