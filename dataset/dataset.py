import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset


class ContrailsDataset(Dataset):
    def __init__(self, df, image_size=256, train=True):
        """Summary of init variables.

        Args:
            df (_type_): Dataframe with IDs and la
            image_size (int, optional): Image size. Defaults to 256.
            train (bool, optional): Train or test dataset. Defaults to True.
        """
        self.df = df
        self.normalize_image = T.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )

        # Preprocess image size
        self.image_size = image_size
        if image_size != 256:
            self.resize_image = T.transforms.Resize(image_size)

        # Not used yet
        self.train = train

    def __getitem__(self, index):
        row = self.df.iloc[index]
        sample_path = row.path
        sample = np.load(str(sample_path))

        img = sample[..., :-1]
        label = sample[..., -1]

        label = torch.tensor(label)

        img = (
            torch.tensor(np.reshape(img, (256, 256, 3)))
            .to(torch.float32)
            .permute(2, 0, 1)
        )

        if self.image_size != 256:
            img = self.resize_image(img)

        img = self.normalize_image(img)

        return img.float(), label.float()

    def __len__(self):
        return len(self.df)
