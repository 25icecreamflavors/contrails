import segmentation_models_pytorch as smp
import timm
import torch.nn as nn

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
            encoder_weights=config["encoder_weights"],
            encoder_depth=config["encoder_depth"],
            decoder_channels=(256, 128, 64, 32, 16)[: config["encoder_depth"]],
            in_channels=3,
            classes=1,
            activation=None,
        )

    def forward(self, batch):
        imgs = batch
        preds = self.model(imgs)
        return preds


class ClassificationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = timm.create_model(
            config["model_name"],
            pretrained=config["pretrained"],
            num_classes=1,  # Binary classification
            in_chans=3,
        )

    def forward(self, x):
        logits = self.model(x)
        return logits
