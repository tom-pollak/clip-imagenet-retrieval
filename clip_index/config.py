from abc import ABCMeta, abstractmethod
import inspect
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
from typing import Any, Callable
import copy

import torch

from clip_index.utils.models import IMAGE_MODELS, TEXT_MODELS

CLIP_PREFIX = "clip:"


@dataclass
class Database:
    conn: sqlite3.Connection | None = None

    def connect_db(self, db_path: Path | str):
        assert Path(db_path).exists(), "Database does not exist"
        self.conn = sqlite3.connect(db_path)

    def db_cursor(self):
        # assert self.conn is not None
        if self.conn is None:
            return None
        return self.conn.cursor()

    def db_commit(self):
        assert self.conn is not None
        self.conn.commit()


@dataclass
class IndexCfg(Database, metaclass=ABCMeta):
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
    include_distances: bool = True

    def load_model(self, device="cpu") -> Callable:
        if self.model_name.startswith(CLIP_PREFIX):
            import clip

            model, preprocess = clip.load(
                self.model_name.replace(CLIP_PREFIX, ""), device=device
            )
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
        if self.model_name.startswith(CLIP_PREFIX):
            import clip

            model, preprocess = clip.load(
                self.model_name.replace(CLIP_PREFIX, ""), jit=False, device=device
            )
            return model.encode_image
        else:
            model = torch.jit.load(  # pyright: ignore [reportPrivateImportUsage]
                str(IMAGE_MODELS[self.model_name]), map_location="cpu"
            )
            return model
