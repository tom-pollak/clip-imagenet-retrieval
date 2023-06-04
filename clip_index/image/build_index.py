from pathlib import Path
import time
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

class ToDevice:
    def __init__(self, device: str):
        self.device = device

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.device)


def _new_axis(t: torch.Tensor):
    return t[np.newaxis, :]


def _convert_image_to_rgb(image):
    return image.convert("RGB")


# this is from CLIP
# https://github.com/openai/CLIP/blob/main/clip/clip.py
# MoTis uses same transformations
def _transform(n_px: int, device: str) -> Callable[[str], torch.Tensor]:
    return Compose(
        [
            Image.open,
            Resize(
                n_px, interpolation=BICUBIC  # pyright: ignore [reportGeneralTypeIssues]
            ),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            ToDevice(device),
            # So we can pass multiple images to image_encoder along a new axis
            _new_axis,
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


@torch.no_grad()
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


@torch.no_grad()
def create_image_embeddings(
    model, image_paths: list[str], input_resolution=224, embedding_size=512, device="cpu"
) -> tuple[torch.Tensor, int]:
    preprocessor = _transform(input_resolution, device)
    image_embeddings = torch.empty((len(image_paths), embedding_size), device=device)
    embedding_time = 0
    for i, path in enumerate(image_paths):
        img = preprocessor(path)
        start_emb_time = time.perf_counter()
        emb = model(img)
        embedding_time += time.perf_counter() - start_emb_time

        image_embeddings[i, :] = emb
    return image_embeddings, embedding_time
