from dataclasses import dataclass
from pathlib import Path
import torch
import math
import faiss
from clip_index import index
from clip_index import config


def norm_tensor(v):
    vnumpy = v.unsqueeze(0).numpy()
    faiss.normalize_L2(vnumpy)
    return vnumpy


def cosine_similarity_to_angular_distance(cos_sim):
    return math.acos(cos_sim)


@dataclass
class HnswImage(index.AnnImage):
    pass


class HnswBuildIndex(index.AnnBuildIndex):
    def __init__(self, cfg):  # : HnswBuildConfig
        self._index = faiss.IndexHNSWFlat(
            cfg.dimension, cfg.M, faiss.METRIC_INNER_PRODUCT
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


class HnswQueryIndex(index.AnnQueryIndex):
    def __init__(self, cfg):  # : AnnoyQueryCfg,  # pyright: ignore
        self._index: faiss.IndexHNSWFlat = faiss.IndexHNSWFlat(
            cfg.dimension, faiss.METRIC_INNER_PRODUCT
        )
        self._cfg = cfg

    def query(
        self,
        query: str,
        qemb: torch.Tensor,
        index_id,
    ) -> list[HnswImage]:
        norm_qemb = norm_tensor(qemb)
        distances, indices = self._index.search(  # type: ignore
            norm_qemb, self._cfg.max_results_per_query
        )
        # angular_distances = [
        #     cosine_similarity_to_angular_distance(d)
        #     for d in distances.squeeze().tolist()
        # ]
        return [
            HnswImage(
                query=query,
                index_id=index_id,
                ref_id=i,
                dist=cosine_similarity_to_angular_distance(d),
                # dist=d
            )
            for i, d in zip(indices.squeeze().tolist(), distances.squeeze().tolist())
            if d <= self._cfg.threshold and i != -1
        ]

    def load(self, path: Path):
        assert path.exists(), f"Index path does not exist: {path}"
        self._index = faiss.read_index(str(path))

    def unload(self):
        self._index = None


@dataclass
class HnswConfig:
    pass


@dataclass
class HnswBuildConfig(config.BuildCfg, HnswConfig):
    M = 32
    
    def __init__(self, M):
        self.M = M

    def load_index(self) -> HnswBuildIndex:
        return HnswBuildIndex(self)


@dataclass
class HnswQueryConfig(config.QueryCfg, HnswConfig):
    def load_index(self) -> HnswQueryIndex:
        return HnswQueryIndex(self)
