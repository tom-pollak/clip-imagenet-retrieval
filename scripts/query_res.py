import os
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import torch

from clip_index.imagenet import ImagenetDETDataset
from clip_index.utils import AnnoyImage
from clip_index.utils import create_index
from clip_index.config import AnnoyQueryCfg

sys.path.append("..")

imagenet_root = Path("/Volumes/T7/ILSVRC/")
imagenet_synset = Path("../assets/imagenet_synset_det.txt")
dataset = ImagenetDETDataset(imagenet_root, imagenet_synset)
queries = [cls for cls in dataset._synset2desc.values()]

image_dir = Path("/Volumes/T7/ILSVRC/Data/DET/val")
image_paths = [image_dir / file for file in os.listdir(image_dir)]

query_embeddings = torch.load("../assets/query_embeddings.pt")


class IndexQueries(ABC):
    keep_topn: int | None = 5

    @abstractmethod
    def __init__(self):
        self.data = ...
        self.stats: dict[str, float] | None = None

    @abstractmethod
    def update(self, ann_img: AnnoyImage):
        ...

    def update_topn(self, topn: list[AnnoyImage], ann_img: AnnoyImage):
        if (len(topn) < 5 or ann_img.dist < topn[-1].dist) and ann_img not in topn:
            topn.append(ann_img)
            topn.sort(key=lambda x: x.dist)
            if self.keep_topn is not None:
                del topn[self.keep_topn :]

    def update_stats(
        self,
        total_time,
        total_load_time,
        total_query_time,
        load_time_per_index,
        query_time_per_query,
    ):
        self.stats = {
            "total_time": total_time,
            "total_load_time": total_load_time,
            "total_query_time": total_query_time,
            "load_time_per_index": load_time_per_index,
            "query_time_per_query": query_time_per_query,
        }


# TODO: Indexwise IndexQueries


class QuerywiseResult(IndexQueries):
    """
    Uses globals: queries
    """

    def __init__(self):
        self.data: dict[str, list[AnnoyImage]] = {q: [] for q in queries}

    def update(self, ann_img: AnnoyImage):
        cur_topn = self.data[ann_img.query]
        self.update_topn(cur_topn, ann_img)


class ImagewiseResult(IndexQueries):
    """
    Uses globals: image_paths
    """

    def __init__(self):
        self.data: list[list[AnnoyImage]] = [[] for _ in range(len(image_paths))]

    def update(self, ann_img: AnnoyImage):
        cur_topn = self.data[ann_img.image_id]
        self.update_topn(cur_topn, ann_img)


def query_index(
    index_queries: IndexQueries,
    index_paths: list[str],
    index_size: int,
    cfg: AnnoyQueryCfg,
) -> IndexQueries:
    """
    Queries indeses at index_paths and updates index_queries in-place
    Uses globals: queries, query_embeddings
    """
    index = create_index()
    # image: [AnnoyImage] (top 5)
    start_time = time.perf_counter()
    total_load_time = 0
    total_query_time = 0
    for path in index_paths:
        index_id = int(Path(path).stem)
        start_load_time = time.perf_counter()
        index.load(path)
        total_load_time += time.perf_counter() - start_load_time
        for query, qemb in zip(queries, query_embeddings):
            start_query_time = time.perf_counter()
            index_ref_ids, distances = index.get_nns_by_vector(
                qemb,  # pyright: reportGeneralTypeIssues=false
                cfg.max_results_per_query,
                include_distances=True,
                search_k=cfg.search_k,
            )
            total_query_time += time.perf_counter() - start_query_time
            for ref_id, dist in zip(index_ref_ids, distances):
                image_id = index_id * index_size + ref_id
                ann_img = AnnoyImage(
                    query=query,
                    ref_id=ref_id,
                    index_id=index_id,
                    dist=dist,
                    image_id=image_id,
                    image_path=image_paths[image_id],
                )
                # dataset.add_imagenet_classes_image(  # NOTE: I think this might be slow
                #     ann_img
                # )
                index_queries.update(ann_img)
        index.unload()
    finish_time = time.perf_counter()
    total_epoch_time = finish_time - start_time
    load_time_per_index = total_load_time / len(index_paths)
    query_time_per_query = total_query_time / len(queries)
    index_queries.update_stats(
        total_epoch_time,
        total_load_time,
        total_query_time,
        load_time_per_index,
        query_time_per_query,
    )
    print(f"\tIndex size {index_size} took {finish_time - start_time:.5f} seconds")
    print(
        f"\tTotal load time: {total_load_time:.5f} seconds, {load_time_per_index:.5f} per index"
    )
    print(
        f"\tTotal query time: {total_query_time:.5f}, {query_time_per_query:.5f} per query"
    )
    print()
    return index_queries


def evaluate_queries(
    index_query_cls: Callable,
    index_folders: list[Path],
    cfg: AnnoyQueryCfg = AnnoyQueryCfg(),
) -> dict[int, ImagewiseResult | QuerywiseResult]:
    print("Evaluating queries, max results per query:", cfg.max_results_per_query)
    result = {}
    for index_folder in index_folders:
        print(f"Starting index size: {index_folder.stem}...")
        index_size = int(index_folder.stem)
        index_paths = [
            str(index_folder / file)
            for file in os.listdir(index_folder)
            if file.endswith(".ann")
        ]
        result[index_size] = query_index(
            index_query_cls(), index_paths, index_size, cfg
        )
    return result


def result_to_dict(
    result: dict[int, ImagewiseResult | QuerywiseResult]
) -> dict[int, dict]:
    return {k: {"data": v.data, "stats": v.stats} for k, v in result.items()}
