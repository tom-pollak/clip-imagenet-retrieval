from pathlib import Path

from clip_index.utils import ROOTDIR


TEXT_MODELS = {
    "motis": Path(ROOTDIR / "model/motis/final_text_encoder_4.pt"),
}

IMAGE_MODELS = {
    "motis": Path(ROOTDIR / "model/motis/final_visual.pt"),
    "mobilevitv2": Path(ROOTDIR / "model/mobilevit/mobilevitv2-2.0.pt"),
    "clip-vit": Path(ROOTDIR / "model/clip-vit/vit_base_16_ft_in1k.pt"),
    "clip:openai-clip-vit-b-32": Path(ROOTDIR / "model/openai/ViT-B-32.pt"),
}
