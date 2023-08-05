import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

AUGMENTATIONS_TRAIN = (
    (A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.2), p=0.5),),
    (
        A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.2), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
    ),
    (
        A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.2), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        A.CoarseDropout(
            max_holes=12,
            max_height=32,
            max_width=32,
            min_holes=6,
            min_height=32,
            min_width=32,
            fill_value=0,
            mask_fill_value=0,
            p=0.5,
        ),
    ),
    (
        A.Flip(p=0.5),
        A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.2), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        A.CoarseDropout(
            max_holes=12,
            max_height=32,
            max_width=32,
            min_holes=6,
            min_height=32,
            min_width=32,
            fill_value=0,
            mask_fill_value=0,
            p=0.5,
        ),
    ),
)


def get_transforms(image_size=512, augmentations=None):
    transforms_resize_list = [
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0
        ),
        ToTensorV2(),
    ]

    if augmentations is not None:
        transforms = A.Compose([*augmentations, *transforms_resize_list])
    else:
        transforms = A.Compose(transforms_resize_list)

    return transforms
