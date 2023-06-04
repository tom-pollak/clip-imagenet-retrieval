import math
from dataclasses import dataclass
from pathlib import Path
import torch
import faiss
from clip_index import index
from clip_index import config


@dataclass
class FlatImage(index.AnnImage):
    pass


def norm_tensor(v):
    vnumpy = v.unsqueeze(0).numpy()
    faiss.normalize_L2(vnumpy)
    return vnumpy


def cosine_similarity_to_angular_distance(cos_sim):
    return math.acos(cos_sim)


class FlatBuildIndex(index.AnnBuildIndex):
    def __init__(self, cfg):  # : FlagBuildCfg,  # pyright: ignore
        # inner product on normalized vector == cosine similarity
        self._index = faiss.index_factory(
            cfg.dimension, "Flat", faiss.METRIC_INNER_PRODUCT
        )
        self._cfg = cfg
        self.build_path = None

    def set_build_path(self, path: Path):
        assert path.parent.exists()
        self.build_path = path

    def add_items(self, tensors: torch.Tensor):
        for i, t in enumerate(tensors):
            assert (
                self._cfg.dimension == t.shape[0]
            ), f"Incompatiable data shape: needed: {self._cfg.dimension}, found: {t.shape}"
            norm_t = norm_tensor(t)
            self._index.add(norm_t)

    def build(self):
        assert self.build_path is not None
        faiss.write_index(self._index, str(self.build_path))


class FlatQueryIndex(index.AnnQueryIndex):
    def __init__(self, cfg):  # : AnnoyQueryCfg,  # pyright: ignore
        self._index = faiss.index_factory(
            cfg.dimension, "Flat", faiss.METRIC_INNER_PRODUCT
        )
        self._cfg = cfg

    def query(
        self,
        query: str,
        qemb: torch.Tensor,
        index_id,
    ) -> list[FlatImage]:
        norm_qemb = norm_tensor(qemb)
        distances, indices = self._index.search(
            norm_qemb, self._cfg.max_results_per_query
        )
        angular_distances = [
            cosine_similarity_to_angular_distance(d)
            for d in distances.squeeze().tolist()
        ]
        return [
            FlatImage(
                query=query,
                ref_id=i,
                index_id=index_id,
                dist=d,
            )
            for i, d in zip(indices.squeeze().tolist(), angular_distances)
            if d <= self._cfg.threshold
        ]

    def load(self, path: Path):
        assert path.exists(), f"Index path does not exist: {path}"
        self._index = faiss.read_index(str(path))

    def unload(self):
        self._index = None


@dataclass
class FlatConfig:
    pass


@dataclass
class FlatBuildConfig(config.BuildCfg, FlatConfig):
    def load_index(self) -> FlatBuildIndex:
        return FlatBuildIndex(self)


@dataclass
class FlatQueryConfig(config.QueryCfg, FlatConfig):
    def load_index(self) -> FlatQueryIndex:
        return FlatQueryIndex(self)
