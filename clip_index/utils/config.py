import abc
import inspect
from dataclasses import dataclass
from typing import Callable, Literal

import torch

from clip_index.utils.nn_utils import load_clip_image, load_clip_text


@dataclass
class BaseCfg(abc.ABC):
    """These cfg dataclasses will be used to adjust the hyperparameters of the model for analysis"""

    model_name: str = "motis"
    pretrained: str | None = None  # For open clip model

    @abc.abstractmethod
    def load_model(self) -> Callable:
        ...

    @classmethod
    def from_dict(cls, env):
        return cls(
            **{k: v for k, v in env.items() if k in inspect.signature(cls).parameters}
        )


@dataclass
class BuildCfg(BaseCfg):
    index_size: int | None = None  # No limit, limits the upper bound of memory usage
    ntrees: int | None = None  # None -> index.f * 2
    vector_size: int = 512
    dist_metric: Literal[
        "angular", "euclidean", "manhattan", "hamming", "dot"
    ] = "angular"
    image_resolution: int = 224

    def load_model(self):
        return load_clip_image(self.model_name, self.pretrained)


@dataclass
class QueryCfg(BaseCfg):
    max_results_per_query: int = 1000  # Seems to be the max
    threshold: float = torch.inf  # Threshold distance for results
    search_k: int = -1  # TODO

    def load_model(self):
        return load_clip_text(self.model_name, self.pretrained)
