import argparse
import gc
import logging
import segmentation_models_pytorch as smp
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.segment_model import SegModel
from train.train_seg import train_model

from utils.setup import read_config, setup_logging, seed_everything
from utils.data import (
    get_dataframes,
    get_folds,
    get_fold_dataloaders,
    get_synthetic_dataloader,
    get_bgs_list,
)


def train(config, model, train_dataloader, valid_dataloader, fold, num_epochs=None):
    if num_epochs is None:
        num_epochs = config["num_epochs"]

    # Instantiate the SegModel and optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["optimizer_params"]["lr"]
    )

    # Create a CosineAnnealingLR scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"])

    # Instantiate the DiceLoss
    dice_loss = smp.losses.DiceLoss(
        mode="binary", from_logits=True, smooth=config["loss_smooth"]
    )

    # Train the model
    model = train_model(
        model,
        train_dataloader,
        valid_dataloader,
        criterion=dice_loss,
        optimizer=optimizer,
        config=config,
        fold=fold,
        scheduler=scheduler,
        num_epochs=num_epochs,
    )
    return model


def main(args):
    """Main script

    Args:
        args (argparse.Namespace): arguments to run the script.
    """

    # Access the values of the arguments
    config_file = args.config
    mode = args.mode

    # Read config file
    config = read_config(config_file)

    # Set up logging messages
    setup_logging(config["name"])
    logging.info("Started the program.")

    # Enable garbage collector and seed everything
    gc.enable()
    seed_everything(config["seed"])

    # Run the train part
    if mode == "train":
        # Get data paths
        df = get_dataframes(config)

        # Get data folds for train and validation
        # If the random splitting is chosen
        df = get_folds(config, df)

        # Train on the selected folds
        for fold in config["train_folds"]:
            logging.info("Started training on - Fold %s", fold)

            # Get train and validation dataloaders
            train_dataloader, valid_dataloader = get_fold_dataloaders(config, df, fold)
            model = SegModel(config)

            if config["synthetic"]["enable"]:
                bgs_list = get_bgs_list("./bgs/")
                synthetic_dataloader = get_synthetic_dataloader(config, bgs_list[:-500])
                synthetic_val_dataloader = get_synthetic_dataloader(
                    config, bgs_list[500:]
                )

                model = train(
                    config,
                    model,
                    synthetic_dataloader,
                    synthetic_val_dataloader,
                    -fold,
                    num_epochs=config["synthetic"]["num_epochs"],
                )

            train(config, model, train_dataloader, valid_dataloader, fold)
            torch.cuda.empty_cache()
            gc.collect()

    # Run the inference part
    elif mode == "inference":
        pass

    logging.info("Finished the program.")


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Training script with YAML config.")
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference"],
        default="train",
        help="Mode: train or inference",
    )
    # Parse the command-line arguments
    args = parser.parse_args()
    # Run main script with arguments
    main(args)
