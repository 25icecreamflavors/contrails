import logging

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


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


if __name__ == "__main__":
    # Set up logging messages
    setup_logging()
    logging.info("Started.")

    # Instantiate the DiceLoss
    dice_loss = smp.losses.DiceLoss(
        mode="binary", from_logits=True, smooth=config["loss_smooth"]
    )

    logging.info("Finished.")
