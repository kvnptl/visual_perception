from tqdm import tqdm
import torch
# from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple
import config
import utils
import os
import glob


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               scaler) -> Tuple[float, float]:
    """
    Trains a PyTorch model for a single epoch.
    """
    model.train()

    train_loss, train_acc, train_dice_score, train_iou_score = 0, 0, 0, 0
    acc = {}

    tqdm_loop = tqdm(dataloader, desc="Train")

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(tqdm_loop):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        # loss.backward()
        scaler.scale(loss).backward()

        # 5. Optimizer step
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        # Calculate and accumulate accuracy metric across all batches
        train_acc += utils.pixel_accuracy(y_pred, y)
        train_dice_score += utils.dice_score_fn(y_pred, y)
        train_iou_score += utils.iou_score_fn(y_pred, y)

        # Update progress bar with training metrics
        tqdm_loop.set_postfix(
            loss=train_loss / (batch + 1),
            acc=train_acc / (batch + 1),
            dice_score=train_dice_score / (batch + 1),
            iou_score=train_iou_score / (batch + 1)
        )

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    acc = {"train_acc": train_acc / len(dataloader),
           "train_dice_score": train_dice_score / len(dataloader),
           "train_iou_score": train_iou_score / len(dataloader)}

    return train_loss, acc


def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """
    Evaluates a PyTorch model for a single epoch.
    """
    model.eval()

    tqdm_loop = tqdm(dataloader, desc="Val")

    val_loss, val_acc, val_dice_score, val_iou_score = 0, 0, 0, 0
    acc = {}

    # Turn on inference context manager
    with torch.inference_mode():
        for batch, (X, y) in enumerate(tqdm_loop):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()

            # Calculate and accumulate accuracy metric across all batches
            val_acc += utils.pixel_accuracy(y_pred, y)
            val_dice_score += utils.dice_score_fn(y_pred, y)
            val_iou_score += utils.iou_score_fn(y_pred, y)

            # Update progress bar with validation metrics
            tqdm_loop.set_postfix(
                loss=val_loss / (batch + 1),
                acc=val_acc / (batch + 1),
                dice_score=val_dice_score / (batch + 1),
                iou_score=val_iou_score / (batch + 1)
            )

    # Adjust metrics to get average loss and accuracy per batch
    val_loss = val_loss / len(dataloader)
    acc = {"val_acc": val_acc / len(dataloader),
           "val_dice_score": val_dice_score / len(dataloader),
           "val_iou_score": val_iou_score / len(dataloader)}

    return val_loss, acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          writer,
          scaler,
          save_model: bool) -> Dict[str, List[float]]:

    # Create empty result dict
    result = {"train_loss": [],
              "train_acc": [],
              "train_dice_score": [],
              "train_iou_score": [],
              "val_loss": [],
              "val_acc": [],
              "val_dice_score": [],
              "val_iou_score": []}

    test_acc_threshold = 0.4
    timestamp = config.TIMESTAMP

    tqdm_loop = tqdm(range(epochs), desc="Epoch")

    if save_model:
        model_save_path = os.path.join(
            config.PARENT_DIR, "results", config.DATASET_NAME, timestamp, "model")
        os.makedirs(model_save_path, exist_ok=True)

    # Loop through training steps
    for epoch in tqdm(tqdm_loop):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device,
                                           scaler=scaler)
        val_loss, val_acc = val_step(model=model,
                                     dataloader=test_dataloader,
                                     loss_fn=loss_fn,
                                     device=device)

        # Print out what's happening
        # print(
        #     f"\n\nEpoch: {epoch+1} | "
        #     f"train_loss: {train_loss:.4f} | "
        #     f"train_acc: {train_acc} | "
        #     # f"train_dice_score: {train_acc['train_dice_score']:.4f} | "
        #     # f"train_iou_score: {train_acc['train_iou_score']:.4f} | "
        #     f"val_loss: {val_loss:.4f} | "
        #     f"val_acc: {val_acc} | "
        #     # f"val_dice_score: {val_acc['val_dice_score']:.4f} | "
        #     # f"val_iou_score: {val_acc['val_iou_score']:.4f}"
        #     f"\n\n"
        # )

        # Update results dictionary
        result["train_loss"].append(train_loss)
        result["train_acc"].append(train_acc["train_acc"])
        result["train_dice_score"].append(train_acc["train_dice_score"])
        result["train_iou_score"].append(train_acc["train_iou_score"])
        result["val_loss"].append(val_loss)
        result["val_acc"].append(val_acc["val_acc"])
        result["val_dice_score"].append(val_acc["val_dice_score"])
        result["val_iou_score"].append(val_acc["val_iou_score"])

        # Write to TensorBoard
        if writer:
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"train_loss": train_loss,
                                 "val_loss": val_loss},
                global_step=epoch
            )
            writer.add_scalars(
                main_tag="Accuracy",
                tag_scalar_dict={"train_acc": train_acc["train_acc"],
                                 "train_dice_score": train_acc["train_dice_score"],
                                 "train_iou_score": train_acc["train_iou_score"],
                                 "val_acc": val_acc["val_acc"],
                                 "val_dice_score": val_acc["val_dice_score"],
                                 "val_iou_score": val_acc["val_iou_score"]},
                global_step=epoch
            )

        # Save model if it's the best yet
        if save_model:
            if val_acc["val_acc"] > test_acc_threshold:
                try:
                    # remove previous best model
                    prev_model = glob.glob(os.path.join(
                        model_save_path, "best_model_*.pth"))[0]
                    if os.path.exists(prev_model):
                        os.remove(prev_model)
                    else:
                        pass
                except:
                    pass

                utils.save_model(model=model,
                                 target_dir=model_save_path,
                                 model_name=f"best_model_{epoch+1}.pth")

                test_acc_threshold = val_acc["val_acc"]

    if writer:
        writer.close()

    return result
