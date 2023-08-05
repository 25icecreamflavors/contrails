import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import cv2


class ContrailsDataset(Dataset):
    def __init__(self, df, train=True, transforms=None):
        """Summary of init variables.

        Args:
            df (_type_): Dataframe with IDs and la
            train (bool, optional): Train or test dataset. Defaults to True.
        """
        self.df = df
        self.train = train
        self.transforms = transforms

    def __getitem__(self, index):
        row = self.df.iloc[index]
        sample_path = row.path
        sample = np.load(str(sample_path))

        item = {
            "image": sample[..., :-1].astype(np.float32),
            "mask": sample[..., -1].astype(np.float32),
        }

        if self.transforms is not None:
            item = self.transforms(**item)
        item["mask"] = item["mask"].unsqueeze(0)

        return item["image"], item["mask"]

    def __len__(self):
        return len(self.df)


class SyntheticContrailsDataset(Dataset):
    def __init__(
        self, images_list, img_size=512, vis_range=(0.7, 0.8), lines_range=(1, 7)
    ):
        self.images_list = images_list
        self.img_size = img_size
        self.vis_range = vis_range
        self.lines_range = lines_range
        self.normalize_image = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    @staticmethod
    def _generate_trails(*, img_size, lines, line_width, median_blur):
        img = np.zeros((img_size, img_size), np.uint8)
        for i in range(lines):
            p = np.random.randint(low=0, high=img_size, size=4)
            cv2.line(img, p[:2], p[2:], 255, line_width)

        mask = np.random.random((img_size, img_size)) < 0.4
        img = mask * (img != 0)
        mask = np.random.random((img_size, img_size)) < 0.05
        img[mask] = True

        img = img.astype(np.uint8) * 255
        img = cv2.medianBlur(img, median_blur)

        lines_det = cv2.HoughLinesP(
            img, 1, np.pi / 180, 10, minLineLength=50, maxLineGap=30
        )
        mask_det = np.zeros((img_size, img_size), np.uint8)
        for line in lines_det:
            for x1, y1, x2, y2 in line:
                cv2.line(mask_det, (x1, y1), (x2, y2), 255, 2)

        img = np.stack([img] * 3, axis=-1)

        return mask_det

    @staticmethod
    def _merge_trails(*, img_bg, trails_mask, img_size, visibility=0.75):
        r_mean, g_mean, b_mean = 0.015702566, 0.24973284, 0.5280043
        r_std, g_std, b_std = 0.024533113, 0.087635785, 0.07859881

        img_bg = cv2.resize(img_bg, (img_size, img_size))
        img_bg = cv2.cvtColor(img_bg, cv2.COLOR_RGB2RGBA)

        img_bg_trsp = img_bg.copy()
        img_bg_trsp[:, :, 3] = 0.0

        blue = np.stack(
            [
                np.random.normal(loc=r_mean, scale=r_std, size=(img_size, img_size)),
                np.random.normal(loc=g_mean, scale=g_std, size=(img_size, img_size)),
                np.random.normal(loc=b_mean, scale=b_std, size=(img_size, img_size)),
                np.ones((img_size, img_size)),
            ],
            axis=-1,
        )
        blue = cv2.blur(blue, (5, 5))

        mask = np.random.random((img_size, img_size)) < visibility
        kernel = np.ones((5, 5), np.uint8)
        mask = mask * cv2.dilate(trails_mask, kernel).astype(bool)
        mask = np.stack([mask] * 4, axis=-1)

        img_bg_trsp[mask] = (img_bg * mask)[mask]
        img_bg_trsp[mask] = blue[mask]
        img_bg_trsp = cv2.blur(img_bg_trsp, (12, 12))

        alpha_img = img_bg_trsp[:, :, 3]

        img_bg[:, :, :3] = img_bg[:, :, :3] * (
            1 - np.stack([alpha_img] * 3, axis=-1)
        ) + img_bg_trsp[:, :, :3] * (np.stack([alpha_img] * 3, axis=-1))

        return img_bg

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_bg = cv2.imread(self.images_list[idx])
        img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB)
        img_bg = img_bg.astype(np.float32) / 255.0

        lines = np.random.randint(*self.lines_range)
        visibility = np.random.uniform(*self.vis_range)

        mask = self._generate_trails(
            img_size=self.img_size, lines=lines, line_width=7, median_blur=3
        )
        img = self._merge_trails(
            img_bg=img_bg,
            trails_mask=mask,
            img_size=self.img_size,
            visibility=visibility,
        )

        img = self.normalize_image(img)
        return img, mask
