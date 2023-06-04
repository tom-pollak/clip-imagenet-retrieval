from pathlib import Path

from clip_index.utils import ROOTDIR


TEXT_MODELS = {
    "motis": Path(ROOTDIR / "model/motis/final_text_encoder_4.pt"),
}

IMAGE_MODELS = {
    "motis": Path(ROOTDIR / "model/motis/final_visual.pt"),
    "clip:openai-clip-vit-b-32": Path(ROOTDIR / "model/openai/ViT-B-32.pt"),
    "clip:openai-clip-vit-l-14": Path(ROOTDIR / "model/openai/ViT-L-14.pt"),
}
