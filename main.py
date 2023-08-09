import argparse
import gc
import logging
import os
import random

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import KFold, StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset.dataset import ContrailsDataset
from models.segment_model import ClassificationModel, SegModel
from train.train_class import train_class_model
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


def setup_logging(say_my_name="debug"):
    """Set up logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        handlers=[
            logging.StreamHandler(),  # Output log messages to console
            logging.FileHandler(
                f"logs/{say_my_name}.log"
            ),  # Save log messages to a file
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
    setup_logging(config["name"])
    logging.info("Started the program.")

    # Enable garbage collector and seed everything
    gc.enable()
    SeedEverything(config["seed"])

    # Run the train part
    if mode == "train":
        # Get data paths
        contrails = os.path.join(config["data_path"], "contrails/")
        train_path = os.path.join(config["data_path"], "train_size.csv")
        valid_path = os.path.join(config["data_path"], "valid_size.csv")

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

        # Take small part of the dataset, if debug
        if config["debug"] == 1:
            df = df.sample(
                frac=0.05, random_state=config["seed"], replace=False
            ).reset_index()

        # Get data folds for train and validation
        # If the random splitting is chosen
        if config["fold_strategy"] == "random":
            Fold = KFold(
                shuffle=True,
                n_splits=config["folds"]["n_splits"],
                random_state=config["folds"]["random_state"],
            )
            # Add folds column to the dataframe
            for fold_number, (trn_index, val_index) in enumerate(
                Fold.split(df)
            ):
                df.loc[val_index, "kfold"] = int(fold_number)
            df["kfold"] = df["kfold"].astype(int)

        # If the splitting is based in the masks sizes
        else:
            df["kfold"] = -1

            # Initialize StratifiedKFold
            skf = StratifiedKFold(
                n_splits=config["folds"]["n_splits"],
                shuffle=True,
                random_state=config["folds"]["random_state"],
            )
            # Create a temporary "mask_size_bin" column to handle bins
            # for non-zero sizes
            df["mask_size_bin"] = pd.qcut(
                df[df["mask_size"] > 0]["mask_size"],
                q=config["folds"]["bins"] - 1,
                labels=False,
            )
            # Assign a unique bin for zero-size masks
            df.loc[df["mask_size"] == 0, "mask_size_bin"] = -1

            # Stratify based on the "mask_size_bin" column and
            # assign fold indices
            for fold_number, (train_index, val_index) in enumerate(
                skf.split(df, df["mask_size_bin"])
            ):
                df.loc[val_index, "kfold"] = int(fold_number)

            # Drop the temporary "mask_size_bin" column
            df.drop(columns=["mask_size_bin"], inplace=True)

            # Get labels for empty masks
            if config["add_pseudo"] != True:
                df.loc[df["mask_size"] != 0, "mask_size"] = 1
                logging.info("Added masks labels.")

            # Convert the fold column to integer
            df["kfold"] = df["kfold"].astype(int)

        if config["add_pseudo"] == True:
            dfs = []

            for i in config["frames_list"]:
                train_path2 = os.path.join(
                    config["data_path"], f"train_ph{i}.csv"
                )
                val_path2 = os.path.join(
                    config["data_path"], f"valid_ph{i}.csv"
                )

                temp_df = pd.read_csv(train_path2)
                dfs.append(temp_df)
                temp_df = pd.read_csv(val_path2)
                dfs.append(temp_df)

            pseudo_df = pd.concat(dfs).reset_index()

            logging.info("Added concatenated frames_df. :D")

            pseudo_df = pseudo_df.merge(
                df[["record_id", "kfold"]],
                left_on="id",
                right_on="record_id",
            )
            pseudo_df = pseudo_df.rename(columns={"record_id_x": "record_id"})
            pseudo_df = pseudo_df.drop(columns=["record_id_y", "index"])

        # Train on the selected folds
        for fold in config["train_folds"]:
            logging.info("Started training on - Fold %s", fold)

            # Add pseudo labels frames dataset
            if config["add_pseudo"] == True:
                train_df = df[df.kfold != fold].reset_index(drop=True)
                train_pseudo = pseudo_df[pseudo_df.kfold != fold].reset_index(
                    drop=True
                )
                train_df = pd.concat([train_df, train_pseudo]).reset_index(
                    drop=True
                )
                logging.info("Concatenated train and pseudo frames datasets.")

            else:
                train_df = df[df.kfold != fold].reset_index(drop=True)
            valid_df = df[df.kfold == fold].reset_index(drop=True)

            if config["train_class"] == False:
                # Create an instance of the ContrailsDataset class
                train_dataset = ContrailsDataset(
                    train_df, image_size=config["image_size"], train=True
                )
                valid_dataset = ContrailsDataset(
                    valid_df, image_size=config["image_size"], train=True
                )
            else:
                # Create a dataset for classification training
                train_dataset = ContrailsDataset(
                    train_df, image_size=config["image_size"], train=False
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

            # Create a CosineAnnealingLR scheduler
            if config["scheduler"]["use"] == 1:
                scheduler = CosineAnnealingLR(
                    optimizer, T_max=config["num_epochs"]
                )
            else:
                scheduler = None

            # Get utils for segment model training
            if config["train_class"] == False:
                # Instantiate the SegModel and optimizer
                model = SegModel(config)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=config["optimizer_params"]["lr"]
                )
                # Instantiate the DiceLoss
                dice_loss = smp.losses.DiceLoss(
                    mode="binary",
                    from_logits=True,
                    smooth=config["loss_smooth"],
                )

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

            # Get utils for a classifier training
            else:
                # Instantiate the ClassifierModel and optimizer
                model = ClassificationModel(config)

                # Instantiate BCEWithLogitsLoss and Adam optimizer
                # BCEWithLogitsLoss combines sigmoid and BCE
                bce_loss = nn.BCEWithLogitsLoss()
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=config["optimizer_params"]["lr"]
                )

                train_class_model(
                    model,
                    train_dataloader,
                    valid_dataloader,
                    criterion=bce_loss,
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
