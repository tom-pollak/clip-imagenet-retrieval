from dataclasses import dataclass
import math
import os
import sqlite3
from pathlib import Path
import time
import torch

from tqdm.auto import trange

from clip_index.image.build_db import get_image_ids, get_next_table_id, insert_images_db
from clip_index.image.build_index import create_image_embeddings
from clip_index.config import BuildCfg
from clip_index.index import AnnBuildIndex


@dataclass
class BuildStats:
    index_size: int
    nindexes: int
    tot_index_add_time: float
    tot_build_time: float
    cpu_build_time: float

    def mean_add_time_per_index(self):
        return self.tot_index_add_time / self.nindexes

    def mean_build_time_per_index(self):
        return self.tot_build_time / self.nindexes

    def mean_cpu_build_time_per_index(self):
        return self.cpu_build_time / self.nindexes

    def mean_add_time_per_item(self):
        return self.tot_index_add_time / (self.index_size * self.nindexes)

    def mean_build_time_per_item(self):
        return self.tot_build_time / (self.index_size * self.nindexes)

    def mean_cpu_build_time_per_item(self):
        return self.cpu_build_time / (self.index_size * self.nindexes)

    def __repr__(self) -> str:
        return f"""Built indexes of size {self.index_size} ({self.nindexes} indexes)
        Index add time: {self.tot_index_add_time:.5f}, {self.mean_add_time_per_item():.5f} per image
        Build time: {self.tot_build_time:.5f}, {self.mean_build_time_per_index():.5f} per index
        CPU build time: {self.cpu_build_time:.5f}, {self.mean_add_time_per_index():.5f} per index\n"""


def build_indexes_from_image_folder(
    image_dir: Path,
    index_folder: Path,
    conn: sqlite3.Connection,
    cfg: BuildCfg,
    cache_image_embeddings: torch.Tensor | None = None,
) -> BuildStats:
    """Builds number of Annoy Indexes from folder of images.
    Returns: list of annoy indexes, NOTE: these are only in memory
    """
    assert (
        image_dir.exists()
        and index_folder.parent.exists()
        and not index_folder.exists()
    ), f"image_dir={image_dir.exists()}, index_folder parent={index_folder.parent.exists()}"
    os.mkdir(index_folder)
    img_paths = [str(image_dir / file) for file in os.listdir(image_dir)]
    if cfg.index_size is None:
        index_size = len(img_paths)
    else:
        index_size = cfg.index_size
    cur = conn.cursor()

    model = cfg.load_model()
    index: AnnBuildIndex = cfg.load_index()

    insert_images_db(cur, img_paths)
    mean_time_per_img = 0
    nimgs = len(img_paths)
    nindexes = math.ceil(nimgs / index_size)

    tot_index_add_time = 0
    tot_build_time = 0
    cpu_build_time = 0

    index_pbar = trange(0, nimgs, index_size)

    for i, index_slice in enumerate(index_pbar):
        start_time = time.perf_counter()
        bimg_path = img_paths[index_slice : index_slice + index_size]
        if cache_image_embeddings is not None:
            # NOTE: These _must_ be in the same order as the image_paths
            image_embeddings = cache_image_embeddings[
                index_slice : index_slice + index_size
            ]
        else:
            image_embeddings = create_image_embeddings(
                model, bimg_path, cfg.image_resolution
            )
        next_id = get_next_table_id(cur, "annoy_index")
        index_path = Path(index_folder) / f"{next_id}.ann"

        index.set_build_path(index_path)
        start_add_items = time.perf_counter()
        index.add_items(image_embeddings)
        finish_add_items = time.perf_counter()

        cpu_time_start_build = time.process_time()
        index.build()
        cpu_time_finish_build = time.process_time()
        finish_build = time.perf_counter()

        insert_indexes_db(cur, index_path, bimg_path)

        tot_index_add_time += finish_add_items - start_add_items
        tot_build_time += finish_build - finish_add_items
        cpu_build_time += cpu_time_finish_build - cpu_time_start_build

        time_per_img = (time.perf_counter() - start_time) / len(bimg_path)
        mean_time_per_img = (mean_time_per_img * i + time_per_img) / (i + 1)
        index_pbar.set_description(
            "%d/%d images, %.2fs/img"
            % (index_slice + index_size, nimgs, mean_time_per_img)
        )
    conn.commit()

    build_stats = BuildStats(
        index_size,
        nindexes,
        tot_index_add_time,
        tot_build_time,
        cpu_build_time,
    )

    print(build_stats)
    return build_stats


def insert_indexes_db(cur: sqlite3.Cursor, index_path: Path, bimg_path: list[str]):
    cur.execute("INSERT INTO annoy_index (index_path) VALUES (?)", (str(index_path),))
    index_id = cur.lastrowid

    image_ids = get_image_ids(cur, bimg_path)
    annoy_index_images = [(index_id, j, imid) for j, imid in enumerate(image_ids)]
    cur.executemany(
        "INSERT INTO annoy_index_image (index_id, ref_id, image_id) VALUES (?, ?, ?)",
        annoy_index_images,
    )
