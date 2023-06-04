import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import torch
from annoy import AnnoyIndex

from clip_index import config
from clip_index.annoy import AnnoyImage, AnnoyQueries
from clip_index.index import AnnQueryIndex

from .simple_tokenizer import tokenize


def add_image_path(cur: sqlite3.Cursor, annoy_images: list[AnnoyImage]):
    """adds image path _inplace_."""
    for im in annoy_images:
        image_id = cur.execute(
            "SELECT image_id FROM annoy_index_image WHERE index_id = ? AND ref_id = ?",
            (im.index_id, im.ref_id),
        ).fetchone()
        if image_id is None:
            print("Image id not found annoy image:", im)
            continue
        else:
            image_id = image_id[0]
        image_path = cur.execute(
            "SELECT image_path FROM image WHERE image_id = ?", (image_id,)
        ).fetchone()[0]
        im.add_image_data(image_id, image_path)


def create_query_embeddings(model, queries: list[str], device="cpu") -> torch.Tensor:
    token_ids = tokenize(queries).to(device)
    encoded_text = model(token_ids)
    return encoded_text


def query_index(
    queries: list[str],
    index_folder: Path,
    cfg: config.QueryCfg,
    cur: sqlite3.Cursor | None = None,
) -> AnnoyQueries:
    assert index_folder.exists()
    encoded_text = create_query_embeddings(cfg.load_model(), queries)
    index: AnnQueryIndex = cfg.load_index()
    index_paths = [
        index_folder / file
        for file in os.listdir(index_folder)
        if file.endswith(".ann")
    ]

    ann_queries: AnnoyQueries = {q: [] for q in queries}
    for path in index_paths:
        index_id = int(path.stem)
        index.load(path)
        for q, qemb in zip(queries, encoded_text):
            ann_imgs = index.query(q, qemb, index_id)
            ann_queries[q] += ann_imgs
        index.unload()

        if cur is not None:
            for ann_imgs in ann_queries.values():
                # NOTE: This will only work with annoy images
                assert type(ann_imgs) == list[AnnoyImage]
                add_image_path(cur, ann_imgs) # pyright: ignore
    return ann_queries


def filter_closest_n_results(query_image_dict, n_results):
    distances = sorted([j.dist for i in query_image_dict.values() for j in i])
    if len(distances) > n_results - 1:
        thres_dist = distances[n_results - 1]
        for q, imgs in query_image_dict.items():
            filterd_imgs = list(filter(lambda x: x.dist <= thres_dist, imgs))
            query_image_dict[q] = filterd_imgs


def create_item_comp_dict(
    index: AnnoyIndex, search_k: int = -1
) -> dict[int, AnnoyImage]:
    """
    Create a dictionary of all items in index to distances of all other items.
    """
    item_dict = {}
    for item_id in range(index.get_n_items()):
        res_ids = index.get_nns_by_item(
            item_id, index.get_n_items(), include_distances=True, search_k=search_k
        )
        item_dict[item_id] = [
            AnnoyImage(query=None, ref_id=ind_id, dist=dist)
            for ind_id, dist in zip(*res_ids)
        ]
    return item_dict
