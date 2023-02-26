from abc import ABCMeta, abstractmethod
import inspect
from dataclasses import dataclass
from typing import Any, Callable
import copy

import torch

from clip_index.utils.models import IMAGE_MODELS, TEXT_MODELS

OPEN_CLIP_PREFIX = "openclip:"


@dataclass
class IndexCfg(metaclass=ABCMeta):
    @abstractmethod
    def load_index(self) -> Any:  # -> index.AnnIndex (circular import)
        ...

    @classmethod
    def from_dict(cls, env):
        return cls(
            **{k: v for k, v in env.items() if k in inspect.signature(cls).parameters}
        )

    def copy(self):
        return copy.deepcopy(self)


@dataclass
class ModelCfg(metaclass=ABCMeta):
    dimension: int = 512
    model_name: str = "motis"
    pretrained: str | None = None  # For open clip model

    @abstractmethod
    def load_model(self, device="cpu") -> Callable:
        ...


@dataclass
class QueryCfg(IndexCfg, ModelCfg, metaclass=ABCMeta):
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
class BuildCfg(IndexCfg, ModelCfg, metaclass=ABCMeta):
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
