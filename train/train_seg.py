import logging
import os

import torch
import wandb
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    config,
    fold=0,
    scheduler=None,
    device="cuda",
):
    """Train method.

    Args:
        model (_type_): pass model for training
        train_loader (_type_): as it sais
        val_loader (_type_): same
        criterion (_type_): pass the loss function
        optimizer (_type_): pass optimizer like Adam
        config (_type_): pass the main config with params
        fold (int, optional): validation fold number. Defaults to 0.
        scheduler (_type_, optional): lr scheduler. Defaults to None.
        device (str, optional): GPU or CPU name. Defaults to "cuda".
    """

    # Get params from the config
    num_epochs = config["num_epochs"]
    image_size = config["image_size"]
    save_path = config["save_path"] + config["name"]
    save_strategy = config["save_strategy"]
    if save_strategy == "epoch":
        save_period = config["save_period"]

    # Check, if the models storage directory exists
    os.makedirs(save_path, exist_ok=True)

    # Set up wandb logging
    if config["debug"] == 1:
        project_name = "debug_contrails"
    else:
        project_name = "contrails_corrected"
    wandb.init(
        # Set the project where this run will be logged
        project=project_name,
        # Set up the run name
        name=(
            f"{config['name']}_lr{config['optimizer_params']['lr']}_"
            f"fold_{fold}_epochs{num_epochs}_image_size_{image_size}"
        ),
        # Track hyperparameters and run metadata
        config={
            "learning_rate": config["optimizer_params"]["lr"],
            "epochs": num_epochs,
            "image_size": image_size,
            "fold": fold,
        },
    )

    # Send model to GPU, set up validation score
    model.to(device)
    best_val_loss = 100

    # Initialize model_path variable for saving the best model
    # Will be used to delete older models
    model_path = ""

    # Create progress bars for training and validation
    train_progress = tqdm(total=num_epochs, desc="Training")
    val_progress = tqdm(total=num_epochs, desc="Validation")

    # Create GradScaler instance
    scaler = GradScaler()

    logging.info("Starting model training!")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Train epoch
        train_batch_progress = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        )
        for batch_idx, (images, labels) in enumerate(train_batch_progress):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():  # Enable autocasting for the forward pass
                outputs = model(images)
                if image_size != 256:
                    outputs = torch.nn.functional.interpolate(
                        outputs, size=256, mode="bilinear"
                    )
                loss = criterion(outputs, labels)

            # Use the scaler to scale the loss and perform backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            # Update training batch progress bar with dice loss
            train_batch_progress.set_postfix(
                {"Train Dice Loss": train_loss / (batch_idx + 1)}
            )
            train_batch_progress.update(1)

        # Update training epoch progress bar with average dice loss
        train_progress.set_postfix(
            {"Train Dice Loss": train_loss / len(train_loader)}
        )
        train_progress.update(1)
        train_batch_progress.close()

        # Scheduler step, if it exists
        if scheduler is not None:
            scheduler.step()

        # Validation epoch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_batch_progress = tqdm(val_loader, desc="Validation")
            for batch_idx, (images, labels) in enumerate(val_batch_progress):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if image_size != 256:
                    outputs = torch.nn.functional.interpolate(
                        outputs, size=256, mode="bilinear"
                    )
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Update validation batch progress bar with dice loss
                val_batch_progress.set_postfix(
                    {"Validation Dice Loss": val_loss / (batch_idx + 1)}
                )
                val_batch_progress.update(1)

        val_loss /= len(val_loader)
        train_loss /= len(train_loader)

        # Update validation progress bar with average dice loss
        val_progress.set_postfix({"Validation Dice Loss": val_loss})
        val_progress.update(1)
        val_batch_progress.close()

        # Log epoch scores (logger and wandb)
        logging.info(
            "Epoch %s/%s completed, "
            "Train Dice Loss: %.3f, "
            "Validation Dice Loss: %.3f "
            "Fold number: %s",
            epoch + 1,
            num_epochs,
            train_loss,
            val_loss,
            fold,
        )
        wandb.log(
            {
                "Epoch": epoch,
                "Train Dice Loss": train_loss,
                "Validation Dice Loss": val_loss,
            }
        )

        # Save model checkpoint based on the chosen save_strategy and
        # save_period
        # Saving each _th epoch
        if save_strategy == "epoch":
            if (epoch + 1) % save_period == 0:
                model_name = (
                    f"model_{config['name']}_fold_{fold}_"
                    f"epoch_{epoch + 1}_val_loss_{val_loss:.3f}.pth"
                )

                # Get full model path and save the model
                model_epoch_path = os.path.join(
                    save_path,
                    model_name,
                )
                torch.save(model.state_dict(), model_epoch_path)
                logging.info("Saved the new model.")

        # Saving the best model only
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                model_name = (
                    f"best_model_{config['name']}_"
                    f"fold_{fold}_epoch_{epoch + 1}_val_loss_{val_loss:.3f}.pth"
                )

                # Get full model path and save the model
                model_epoch_path = os.path.join(
                    save_path,
                    model_name,
                )
                torch.save(model.state_dict(), model_epoch_path)
                logging.info("Saved the new best model.")

                # Delete the previous best model
                if os.path.exists(model_path):
                    os.remove(model_path)
                model_path = model_epoch_path

    wandb.finish()
    logging.info("Training complete!")
