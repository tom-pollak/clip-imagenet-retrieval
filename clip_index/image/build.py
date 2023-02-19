import os
import sqlite3
from pathlib import Path
from time import time

from annoy import AnnoyIndex
from tqdm.auto import trange

from clip_index.image.build_db import get_image_ids, get_next_table_id, insert_images_db
from clip_index.image.build_index import add_images_to_index, create_image_embeddings
from clip_index.utils.config import BuildCfg


def build_indexes_from_imgs(
    model,
    image_dir: Path,
    index_folder: Path,
    conn: sqlite3.Connection,
    cfg: BuildCfg = BuildCfg(),
):
    assert image_dir.exists() and index_folder.exists()
    """Builds number of Annoy Indexes from folder of images.
    Returns: list of annoy indexes, NOTE: these are only in memory
    """
    img_paths = [str(image_dir / file) for file in os.listdir(image_dir)]
    if cfg.index_size is None:
        index_size = len(img_paths)
    else:
        index_size = cfg.index_size
    cur = conn.cursor()

    insert_images_db(cur, img_paths)
    mean_time_per_img = 0
    nimgs = len(img_paths)
    index_pbar = trange(0, nimgs, index_size)
    for i, index_slice in enumerate(index_pbar):
        start_time = time()
        bimg_path = img_paths[index_slice : index_slice + index_size]
        index = AnnoyIndex(cfg.vector_size, cfg.dist_metric)

        next_id = get_next_table_id(cur, "annoy_index")
        index_path = Path(index_folder) / f"{next_id}.ann"
        assert not index_path.exists(), f"Index path already exists: {index_path}"

        image_embeddings = create_image_embeddings(
            model, bimg_path, cfg.image_resolution
        )
        index.on_disk_build(str(index_path))
        add_images_to_index(index, image_embeddings)

        if cfg.ntrees:
            ntrees = cfg.ntrees
        else:
            print(f"Number of trees not set, defaulting to {index.f * 2}")
            ntrees = index.f * 2  # 512 * 2

        index.build(ntrees)
        index.unload()

        cur.execute(
            "INSERT INTO annoy_index (index_path) VALUES (?)", (str(index_path),)
        )
        index_id = cur.lastrowid

        image_ids = get_image_ids(cur, bimg_path)
        annoy_index_images = [(index_id, j, imid) for j, imid in enumerate(image_ids)]
        cur.executemany(
            "INSERT INTO annoy_index_image (index_id, ref_id, image_id) VALUES (?, ?, ?)",
            annoy_index_images,
        )
        time_per_img = (time() - start_time) / len(bimg_path)
        mean_time_per_img = (mean_time_per_img * i + time_per_img) / (i + 1)
        index_pbar.set_description(
            "%d/%d images, %.2fs/img"
            % (index_slice + index_size, nimgs, mean_time_per_img)
        )
    conn.commit()
