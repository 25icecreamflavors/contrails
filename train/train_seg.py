import os

import torch
from tqdm import tqdm


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

    model.to(device)
    best_val_loss = 100
    # Initialize model_path variable for saving the best model
    model_path = ""

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Train epoch
        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
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

        # Scheduler step, if it exists
        if scheduler is not None:
            scheduler.step()

        # Validation epoch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if image_size != 256:
                    outputs = torch.nn.functional.interpolate(
                        outputs, size=256, mode="bilinear"
                    )
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Print epoch scores
        print(
            f"Epoch {epoch + 1}/{num_epochs}, \
            Train Dice Loss: {train_loss / len(train_loader)}, \
            Validation Dice Loss: {val_loss}"
        )

        # Save model checkpoint based on the chosen save_strategy and
        # save_period
        # Saving each _th epoch
        if save_strategy == "epoch":
            if (epoch + 1) % save_period == 0:
                model_epoch_path = os.path.join(
                    save_path, f"model_epoch_{epoch + 1}.pth"
                )
                torch.save(model.state_dict(), model_epoch_path)

        # Saving the best model only
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_epoch_path = os.path.join(
                    save_path, f"best_model_epoch_{epoch + 1}.pth"
                )
                torch.save(model.state_dict(), model_epoch_path)

                # Delete the previous best model
                if os.path.exists(model_path):
                    os.remove(model_path)
                model_path = model_epoch_path

    print("Training complete!")
