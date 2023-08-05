import os
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from torch.utils.data import DataLoader
from dataset.dataset import ContrailsDataset, SyntheticContrailsDataset
from dataset.transforms import get_transforms, AUGMENTATIONS_TRAIN


def get_dataframes(config):
    contrails = os.path.join(config["data_path"], "contrails/")
    train_path = os.path.join(config["data_path"], "train_df.csv")
    valid_path = os.path.join(config["data_path"], "valid_df.csv")

    # Read dataframes with contrails id
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)

    # Add contrails paths to dataframes and concatenate them
    train_df["path"] = contrails + train_df["record_id"].astype(str) + ".npy"
    valid_df["path"] = contrails + valid_df["record_id"].astype(str) + ".npy"
    df = pd.concat([train_df, valid_df]).reset_index()

    # Take small part of the dataset, if debug
    if config["debug"] == 1:
        df = df.sample(
            frac=0.05, random_state=config["seed"], replace=False
        ).reset_index()

    return df


def get_folds(config, df):
    if config["fold_strategy"] == "random":
        Fold = KFold(
            shuffle=True,
            n_splits=config["folds"]["n_splits"],
            random_state=config["folds"]["random_state"],
        )
        # Add folds column to the dataframe
        for fold_number, (trn_index, val_index) in enumerate(Fold.split(df)):
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
        # Convert the fold column to integer
        df["kfold"] = df["kfold"].astype(int)

    return df


def get_fold_dataloaders(config, df, fold):
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    if config["augmentations"]["enable"]:
        augs = AUGMENTATIONS_TRAIN[config["augmentations"]["set"]]
    else:
        augs = None
    transforms = get_transforms(config["image_size"], augs)

    # Create an instance of the ContrailsDataset class
    train_dataset = ContrailsDataset(
        train_df,
        train=True,
        transforms=transforms,
    )
    valid_dataset = ContrailsDataset(
        valid_df,
        train=False,
        transforms=transforms,
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

    return train_dataloader, valid_dataloader


def get_synthetic_dataloader(config, bgs_list):
    train_dataset = SyntheticContrailsDataset(
        bgs_list,
        img_size=config["image_size"],
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["train_bs"],
        shuffle=True,
        num_workers=config["num_workers"],
    )

    return train_dataloader


def get_bgs_list(path_dir, extension=".jpg"):
    bgs_list = []
    for path_file in os.listdir(path_dir):
        if path_file.endswith(extension):
            bgs_list.append(os.path.join(path_dir, path_file))
    bgs_list = sorted(bgs_list)
    return bgs_list
