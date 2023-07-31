import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# Instantiate the DiceLoss
dice_loss = smp.losses.DiceLoss(
    mode="binary", from_logits=True, smooth=config["loss_smooth"]
)
