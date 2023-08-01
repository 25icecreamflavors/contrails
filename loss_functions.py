import torch


def dice_loss(preds, targets):
    intersection = torch.sum(preds * targets)
    union = torch.sum(preds) + torch.sum(targets)
    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
    return 1 - dice
