import argparse
import segmentation_models_pytorch as smp


def download_encoder_weights(encoder_name):
    """Download model from timm that is not built in smp.

    Args:
        encoder_name (str): model name from timm
    """

    model = smp.Unet(encoder_name=encoder_name)
    print(f"Downloading pretrained weights for {encoder_name}...")
    model.eval()
    print(f"Pretrained weights for {encoder_name} downloaded successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download pretrained encoder weights for segmentation models."
    )
    parser.add_argument(
        "encoder_names",
        nargs="+",
        type=str,
        help="Names of encoder models to download weights for.",
    )
    args = parser.parse_args()

    for encoder_name in args.encoder_names:
        download_encoder_weights(encoder_name)
