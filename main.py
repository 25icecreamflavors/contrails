import argparse
import gc
import logging
import os
import random

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import yaml
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset.dataset import ContrailsDataset
from models.segment_model import SegModel
from train.train_seg import train_model


def SeedEverything(seed=808):
    """Method to seed everything."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_config(file_path):
    """Open and read yaml config.

    Args:
        file_path (str): path to config file

    Returns:
        dict: config file
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def setup_logging():
    """Set up logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        handlers=[
            logging.StreamHandler(),  # Output log messages to console
            logging.FileHandler("somelogs.log"),  # Save log messages to a file
        ],
    )


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
    setup_logging()
    logging.info("Started the program.")

    # Enable garbage collector and seed everything
    gc.enable()
    SeedEverything(config["seed"])

    # Run the train part
    if mode == "train":
        # Get data paths
        contrails = os.path.join(config["data_path"], "contrails/")
        train_path = os.path.join(config["data_path"], "train_df.csv")
        valid_path = os.path.join(config["data_path"], "valid_df.csv")

        # Read dataframes with contrails id
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)

        # Add contrails paths to dataframes and concatenate them
        train_df["path"] = (
            contrails + train_df["record_id"].astype(str) + ".npy"
        )
        valid_df["path"] = (
            contrails + valid_df["record_id"].astype(str) + ".npy"
        )
        df = pd.concat([train_df, valid_df]).reset_index()

        if config["debug"] == 1:
            df = df.sample(frac=0.05, random_state=config["seed"])

        # Get data folds for train and validation
        Fold = KFold(
            shuffle=True,
            n_splits=config["folds"]["n_splits"],
            random_state=config["folds"]["random_state"],
        )
        # Add folds column to the dataframe
        for fold_number, (trn_index, val_index) in enumerate(Fold.split(df)):
            df.loc[val_index, "kfold"] = int(fold_number)
        df["kfold"] = df["kfold"].astype(int)

        # Train on the selected folds
        for fold in config["train_folds"]:
            logging.info("Started training on - Fold %s", fold)
            train_df = df[df.kfold != fold].reset_index(drop=True)
            valid_df = df[df.kfold == fold].reset_index(drop=True)

            # Create an instance of the ContrailsDataset class
            train_dataset = ContrailsDataset(
                train_df, image_size=config["image_size"], train=True
            )
            valid_dataset = ContrailsDataset(
                valid_df, image_size=config["image_size"], train=False
            )
            # Get dataloaders
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=config["train_bs"],
                shuffle=True,
                num_workers=config["num_workers"],
            )
            valid_dataloader = DataLoader(
                valid_dataset,
                batch_size=config["valid_bs"],
                shuffle=False,
                num_workers=config["num_workers"],
            )

            # Instantiate the SegModel and optimizer
            model = SegModel(config)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=config["optimizer_params"]["lr"]
            )

            # Create a CosineAnnealingLR scheduler
            scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"])

            # Instantiate the DiceLoss
            dice_loss = smp.losses.DiceLoss(
                mode="binary", from_logits=True, smooth=config["loss_smooth"]
            )

            # Train the model
            train_model(
                model,
                train_dataloader,
                valid_dataloader,
                criterion=dice_loss,
                optimizer=optimizer,
                config=config,
                fold=fold,
                scheduler=scheduler,
            )

            # Clear GPU memory
            del model
            torch.cuda.empty_cache()
            gc.collect()

    # Run the inference part
    elif mode == "inference":
        pass

    logging.info("Finished the program.")


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Training script with YAML config."
    )
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
