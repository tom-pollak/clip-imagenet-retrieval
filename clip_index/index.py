from pathlib import Path
import annoy
from abc import ABC, abstractmethod
import torch

from clip_index.text.query import AnnoyImage
from clip_index.utils.config import (
    AnnoyBuildCfg,
    AnnoyCfg,
    BaseCfg,
    BuildCfg,
    QueryCfg,
    AnnoyQueryCfg,
)


class AnnIndex(ABC):
    @abstractmethod
    def __init__(
        self,
        cfg: BaseCfg,
    ):
        ...

    @abstractmethod
    def add_items(self, tensors: list[torch.Tensor]):
        ...

    @abstractmethod
    def build(self, path: Path, build_cfg: BuildCfg):
        ...

    @abstractmethod
    def query(
        self,
        query: str,
        qemb: torch.Tensor,
        query_cfg: QueryCfg,
        index_id: int | None = None,
    ) -> list[AnnoyImage]:
        ...

    @abstractmethod
    def unload(self):
        ...

    @abstractmethod
    def load(self, path: Path):
        ...


class AnnoyIndex(AnnIndex):
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
