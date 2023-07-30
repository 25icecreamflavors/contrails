import torch.nn as nn
import segmentation_models_pytorch as smp

seg_models = {
    "Unet": smp.Unet,
    "Unet++": smp.UnetPlusPlus,
    "MAnet": smp.MAnet,
    "Linknet": smp.Linknet,
    "FPN": smp.FPN,
    "PSPNet": smp.PSPNet,
    "PAN": smp.PAN,
    "DeepLabV3": smp.DeepLabV3,
    "DeepLabV3+": smp.DeepLabV3Plus,
}


class SegModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = seg_models[config["seg_model"]](
            encoder_name=config["encoder_name"],
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )

    def forward(self, batch):
        imgs = batch
        preds = self.model(imgs)
        return preds
