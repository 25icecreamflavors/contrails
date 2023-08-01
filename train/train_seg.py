import logging
import os

import torch
from tqdm import tqdm
import wandb


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    config,
    scheduler=None,
    device="cuda",
):
    # Get params from the config
    num_epochs = config["num_epochs"]
    image_size = config["image_size"]
    save_path = config["save_path"]
    save_strategy = config["save_strategy"]
    if save_strategy == "epoch":
        save_period = config["save_period"]

    # Set up wandb logging
    run = wandb.init(
        # Set the project where this run will be logged
        project=(
            f"{config['name']}_lr{config['optimizer_params']['lr']}_"
            f"epochs{num_epochs}_image_size_{image_size}"
        ),
        # Track hyperparameters and run metadata
        config={
            "learning_rate": config["optimizer_params"]["lr"],
            "epochs": num_epochs,
            "image_size": image_size,
        },
    )

    # Send model to GPU, set up validation score
    model.to(device)
    best_val_loss = 100

    # Initialize model_path variable for saving the best model
    model_path = ""

    # Create progress bars for training and validation
    train_progress = tqdm(total=num_epochs, desc="Training")
    val_progress = tqdm(total=num_epochs, desc="Validation")

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

            outputs = model(images)
            if image_size != 256:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=256, mode="bilinear"
                )

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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
            f"Epoch {epoch + 1}/{num_epochs} completed, \
            Train Dice Loss: {train_loss}, \
            Validation Dice Loss: {val_loss}"
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
                model_epoch_path = os.path.join(
                    save_path, f"model_{config['name']}_epoch_{epoch + 1}.pth"
                )
                torch.save(model.state_dict(), model_epoch_path)
                logging.info("Saved the new model.")

        # Saving the best model only
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_epoch_path = os.path.join(
                    save_path,
                    f"best_model_{config['name']}_epoch_{epoch + 1}.pth",
                )
                torch.save(model.state_dict(), model_epoch_path)
                logging.info("Saved the new best model.")

                # Delete the previous best model
                if os.path.exists(model_path):
                    os.remove(model_path)
                model_path = model_epoch_path

    logging.info("Training complete!")
