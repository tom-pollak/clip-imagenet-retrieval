from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod
from typing import TypeAlias
import torch

import clip_index.config as config


@dataclass
class AnnImage(ABC):
    query: str | None
    dist: float
    image_id: int | None = None
    image_path: Path | None = None
    imagenet_classes: set[str] | None = None
    index_id: int | None = None
    ref_id: int | None = None

    def add_image_data(self, imid: int, path: str):
        self.image_id = imid
        self.image_path = Path(path)
        assert self.image_path.exists()

    def __eq__(self, item) -> bool:
        same_image: bool = (
            self.image_id == item.image_id
            or self.image_path == item.image_path
            # or (self.index_id == item.index_id and self.ref_id == item.ref_id)
        )
        same_index_ref = self.index_id == item.index_id and self.ref_id == item.ref_id
        same_query: bool = self.query == item.query
        return same_image and same_query or same_query or same_index_ref


AnnQueries: TypeAlias = dict[str, list[AnnImage]]


class AnnQueryIndex(ABC):
    @abstractmethod
    def __init__(
        self,
        cfg: config.QueryCfg,
    ):
        ...

    @abstractmethod
    def query(
        self,
        query: str,
        qemb: torch.Tensor,
        index_id: int | None = None,
    ) -> list[AnnImage]:
        ...

    @abstractmethod
    def load(self, path: Path):
        ...

    @abstractmethod
    def unload(self):
        ...


class AnnBuildIndex(ABC):
    @abstractmethod
    def __init__(
        self,
        cfg: config.BuildCfg,
    ):
        ...

    @abstractmethod
    def add_items(self, tensors: torch.Tensor):
        ...

    @abstractmethod
    def build(self):
        ...

    @abstractmethod
    def set_build_path(self, path: Path):
        ...
