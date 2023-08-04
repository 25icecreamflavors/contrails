import random
import numpy as np
import os
import yaml
import logging
import torch


def seed_everything(seed=808):
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
    if not os.path.exists("logs"):
        os.makedirs("logs")
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
