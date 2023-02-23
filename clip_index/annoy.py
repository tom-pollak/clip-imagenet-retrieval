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


AnnoyQueries: TypeAlias = dict[str, list[AnnoyImage]]


@dataclass
class AnnoyCfg(config.BaseCfg, metaclass=ABCMeta):
    dist_metric: Literal[
        "angular", "euclidean", "manhattan", "hamming", "dot"
    ] = "angular"

    def load_index(self) -> Any:  # -> AnnoyIndex
        return AnnoyIndex(self)


@dataclass
class AnnoyBuildCfg(config.BuildCfg, AnnoyCfg):
    ntrees: int | None = None  # None -> index.f * 2

    # def load_index(self) -> AnnoyIndex:
    #     return AnnoyCfg.load_index(self)


@dataclass
class AnnoyQueryCfg(config.QueryCfg, AnnoyCfg):
    search_k: int = -1  # TODO

    # def load_index(self) -> AnnoyIndex:
    #     return AnnoyCfg.load_index(self)


class AnnoyIndex(index.AnnIndex):
    def __init__(
        self,
        cfg: AnnoyCfg,
    ):
        self._cfg = cfg
        self._index = annoy.AnnoyIndex(
            self._cfg.dimension, metric=self._cfg.dist_metric
        )

    def add_items(self, tensors: list[torch.Tensor]):
        for i, t in enumerate(tensors):
            assert (
                self._cfg.dimension == t.shape[0]
            ), f"Incompatiable data shape: needed: {self._cfg.dimension}, found: {t.shape}"
            self._index.add_item(i, t)  # pyright: reportGeneralTypeIssues=false

    def load(self, path: Path):
        assert path.exists(), f"Index path does not exist: {path}"
        self._index.load(str(path))

    def unload(self):
        self._index.unload()

    def build(self, path: Path, build_cfg: AnnoyBuildCfg):
        assert not path.exists(), f"Index path already exists: {path}"
        self._index.on_disk_build(str(path))

        if build_cfg.ntrees is None:
            ntrees = self._index.f * 2  # 512 * 2
        else:
            ntrees = build_cfg.ntrees

        self._index.build(ntrees)

    def query(
        self,
        query: str,
        qemb: torch.Tensor,
        query_cfg: AnnoyQueryCfg,
        index_id: int | None = None,
    ) -> list[AnnoyImage]:
        indices, distances = self._index.get_nns_by_vector(
            qemb.tolist(),
            query_cfg.max_results_per_query,
            search_k=query_cfg.search_k,
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
            if d <= query_cfg.threshold
        ]
