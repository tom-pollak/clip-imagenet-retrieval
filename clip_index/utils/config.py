from abc import ABCMeta, abstractmethod
import inspect
from dataclasses import dataclass
from typing import Callable, Literal

import torch

from pathlib import Path
from .utils import ROOTDIR

from clip_index.index import AnnIndex, AnnoyIndex

TEXT_MODELS = {
    "motis": Path(ROOTDIR / "model/motis/final_text_encoder_4.pt"),
}

IMAGE_MODELS = {
    "motis": Path(ROOTDIR / "model/motis/final_visual.pt"),
    "mobilevitv2": Path(ROOTDIR / "model/mobilevit/mobilevitv2-2.0.pt"),
    "clip-vit": Path(ROOTDIR / "model/clip-vit/vit_base_16_ft_in1k.pt"),
    "clip:openai-clip-vit-b-32": Path(ROOTDIR / "model/openai/ViT-B-32.pt"),
}


OPEN_CLIP_PREFIX = "openclip:"


@dataclass
class BaseCfg(metaclass=ABCMeta):
    """These cfg dataclasses will be used to adjust the hyperparameters of the model for analysis"""

    dimension: int = 512
    model_name: str = "motis"
    pretrained: str | None = None  # For open clip model

    @abstractmethod
    def load_model(self, device="cpu") -> Callable:
        ...

    @abstractmethod
    def load_index(self) -> AnnIndex:
        ...

    @classmethod
    def from_dict(cls, env):
        return cls(
            **{k: v for k, v in env.items() if k in inspect.signature(cls).parameters}
        )


@dataclass
class QueryCfg(BaseCfg, metaclass=ABCMeta):
    threshold: float = torch.inf  # Threshold distance for results
    max_results_per_query: int = 1000  # Seems to be the max

    def load_model(self, device="cpu") -> Callable:
        if self.model_name.startswith(OPEN_CLIP_PREFIX):
            import open_clip

            assert self.pretrained is not None
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name=self.model_name.replace(OPEN_CLIP_PREFIX, ""),
                pretrained=self.pretrained,
            )
            assert isinstance(model, open_clip.CLIP)
            return model.encode_text
        else:
            model = torch.jit.load(  # pyright: ignore [reportPrivateImportUsage]
                str(TEXT_MODELS[self.model_name]), map_location=device
            )
            return model


@dataclass
class BuildCfg(BaseCfg, metaclass=ABCMeta):
    index_size: int | None = None  # No limit, limits the upper bound of memory usage
    image_resolution: int = 224

    def load_model(self, device="cpu") -> Callable:
        if self.model_name.startswith(OPEN_CLIP_PREFIX):
            import open_clip

            assert self.pretrained is not None
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name=self.model_name.replace(OPEN_CLIP_PREFIX, ""),
                pretrained=self.pretrained,
            )
            assert isinstance(model, open_clip.CLIP)
            return model.encode_image
        else:
            model = torch.jit.load(  # pyright: ignore [reportPrivateImportUsage]
                str(IMAGE_MODELS[self.model_name]), map_location=device
            )
            return model


@dataclass
class AnnoyCfg(BaseCfg, metaclass=ABCMeta):
    dist_metric: Literal[
        "angular", "euclidean", "manhattan", "hamming", "dot"
    ] = "angular"

    def load_index(self) -> AnnoyIndex:
        return AnnoyIndex(self)


@dataclass
class AnnoyBuildCfg(BuildCfg, AnnoyCfg):
    ntrees: int | None = None  # None -> index.f * 2

    def load_index(self) -> AnnoyIndex:
        return AnnoyCfg.load_index(self)


@dataclass
class AnnoyQueryCfg(QueryCfg, AnnoyCfg):
    search_k: int = -1  # TODO

    def load_index(self) -> AnnoyIndex:
        return AnnoyCfg.load_index(self)
