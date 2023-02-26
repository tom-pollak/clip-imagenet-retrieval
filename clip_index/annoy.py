from abc import ABCMeta
from dataclasses import dataclass
from typing import Any, TypeAlias
from pathlib import Path
from typing import Literal
import annoy
import torch
from clip_index import index
from clip_index import config


@dataclass
class AnnoyImage(index.AnnImage):
    ref_id: int | None = None
    index_id: int | None = None

    def __eq__(self, item) -> bool:
        same_index_ref = self.index_id == item.index_id and self.ref_id == item.ref_id
        same_query: bool = self.query == item.query
        return super().__eq__(item) or (same_index_ref and same_query)


AnnoyQueries: TypeAlias = dict[str, list[index.AnnImage]]


class AnnoyBuildIndex(index.AnnBuildIndex):
    def __init__(self, cfg):  # : AnnoyBuildCfg,  # pyright: ignore
        self._index = annoy.AnnoyIndex(cfg.dimension, metric=cfg.dist_metric)
        self._cfg = cfg

    def set_build_path(self, path: Path):
        assert path.parent.exists()
        self._index.on_disk_build(str(path))

    def add_items(self, tensors: torch.Tensor):
        for i, t in enumerate(tensors):
            assert (
                self._cfg.dimension == t.shape[0]
            ), f"Incompatiable data shape: needed: {self._cfg.dimension}, found: {t.shape}"
            self._index.add_item(i, t)  # pyright: ignore

    def build(self):
        if self._cfg.ntrees is None:
            ntrees = self._index.f * 2  # 512 * 2
        else:
            ntrees = self._cfg.ntrees

        self._index.build(ntrees)
        self._index.unload()
        # HACK: annoy doesn't seem to be able to load the index after building it
        self._index = annoy.AnnoyIndex(
            self._cfg.dimension, metric=self._cfg.dist_metric
        )


class AnnoyQueryIndex(index.AnnQueryIndex):
    def __init__(self, cfg):  # : AnnoyQueryCfg,  # pyright: ignore
        self._index = annoy.AnnoyIndex(cfg.dimension, metric=cfg.dist_metric)
        self._cfg = cfg

    def query(
        self,
        query: str,
        qemb: torch.Tensor,
        index_id: int | None = None,
    ) -> list[AnnoyImage]:
        indices, distances = self._index.get_nns_by_vector(
            qemb.tolist(),  # pyright: ignore
            self._cfg.max_results_per_query,
            search_k=self._cfg.search_k,
            include_distances=True,
        )
        return [
            AnnoyImage(
                query=query,
                index_id=index_id,
                ref_id=i,
                dist=d,
            )
            for i, d in zip(indices, distances)
            if d <= self._cfg.threshold
        ]

    def load(self, path: Path):
        assert path.exists(), f"Index path does not exist: {path}"
        self._index.load(str(path))

    def unload(self):
        self._index.unload()


@dataclass
class AnnoyCfg:
    dist_metric: Literal[
        "angular", "euclidean", "manhattan", "hamming", "dot"
    ] = "angular"


@dataclass
class AnnoyBuildCfg(config.BuildCfg, AnnoyCfg):
    ntrees: int | None = None  # None -> index.f * 2, ntrees affects build time abit, but mostly query time

    def load_index(self) -> AnnoyBuildIndex:
        return AnnoyBuildIndex(self)


@dataclass
class AnnoyQueryCfg(config.QueryCfg, AnnoyCfg):
    search_k: int = -1  # TODO

    def load_index(self) -> AnnoyQueryIndex:
        return AnnoyQueryIndex(self)
