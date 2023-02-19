from pathlib import Path
from typing import Callable

import torch

from .utils import ROOTDIR

TEXT_MODELS = {
    "motis": Path(ROOTDIR / "model/motis/final_text_encoder_4.pt"),
}

IMAGE_MODELS = {
    "motis": Path(ROOTDIR / "model/motis/final_visual.pt"),
    "mobilevitv2": Path(ROOTDIR / "model/mobilevit/mobilevitv2-2.0.pt"),
    "clip-vit": Path(ROOTDIR / "model/clip-vit/vit_base_16_ft_in1k.pt"),
    "clip:openai-clip-vit-b-32": Path(ROOTDIR / "model/openai/ViT-B-32.pt"),
}


def load_clip_image(
    model_name="motis", pretrained: str | None = None, device="cpu"
) -> Callable:
    if model_name.startswith("openclip:"):
        import open_clip

        assert pretrained is not None
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name.replace("openclip:", ""), pretrained=pretrained
        )
        assert isinstance(model, open_clip.CLIP)
        return model.encode_image
    else:
        model = torch.jit.load(  # pyright: ignore [reportPrivateImportUsage]
            str(IMAGE_MODELS[model_name]), map_location=device
        )
        return model


def load_clip_text(
    model_name="motis", pretrained: str | None = None, device="cpu"
) -> Callable:
    if model_name.startswith("openclip:"):
        import open_clip

        assert pretrained is not None
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name.replace("openclip:", ""), pretrained=pretrained
        )
        assert isinstance(model, open_clip.CLIP)
        return model.encode_text
    else:
        model = torch.jit.load(  # pyright: ignore [reportPrivateImportUsage]
            str(TEXT_MODELS[model_name]), map_location=device
        )
        return model
