from pathlib import Path
from typing import Callable

import numpy as np
import torch
from annoy import AnnoyIndex
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.Resampling.BICUBIC

device = "cuda" if torch.cuda.is_available() else "cpu"


def _new_axis(t: torch.Tensor):
    return t[np.newaxis, :]


def _convert_image_to_rgb(image):
    return image.convert("RGB")


# this is from CLIP
# https://github.com/openai/CLIP/blob/main/clip/clip.py
# MoTis uses same transformations
def _transform(n_px: int) -> Callable[[str], torch.Tensor]:
    return Compose(
        [
            Image.open,
            Resize(
                n_px, interpolation=BICUBIC  # pyright: ignore [reportGeneralTypeIssues]
            ),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            # So we can pass multiple images to image_encoder along a new axis
            _new_axis,
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def add_images_to_index(
    index: AnnoyIndex,
    image_embeddings: list[torch.Tensor],
):
    """
    Loading images, encoding with clip and adding to Annooy index
    """
    for i, img in enumerate(image_embeddings):
        assert img.shape[0] == index.f, f"Incompatiable data shape {img.shape}"
        index.add_item(i, img)  # pyright: ignore


def create_image_embeddings(
    model, image_paths: list[str], input_resolution=224, embedding_size=512
) -> torch.Tensor:
    preprocessor = _transform(input_resolution)
    image_embeddings = torch.empty((len(image_paths), embedding_size))
    for i, path in enumerate(image_paths):
        img = preprocessor(path)
        emb = model(img)
        image_embeddings[i, :] = emb
    return image_embeddings
