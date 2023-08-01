import argparse
import logging

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import yaml


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
    logging.info("Started.")

    # Run the train part
    if mode == "train":
        # Instantiate the DiceLoss
        dice_loss = smp.losses.DiceLoss(
            mode="binary", from_logits=True, smooth=config["loss_smooth"]
        )

    # Run the inference part
    elif mode == "inference":
        pass

    logging.info("Finished.")


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
