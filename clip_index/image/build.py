import os
import sqlite3
from pathlib import Path
from time import time

# from annoy import AnnoyIndex
from clip_index.index import AnnoyIndex
from tqdm.auto import trange

from clip_index.image.build_db import get_image_ids, get_next_table_id, insert_images_db
from clip_index.image.build_index import add_images_to_index, create_image_embeddings
from clip_index.utils.config import AnnoyBuildCfg, BuildCfg


def build_indexes_from_image_folder(
    image_dir: Path, index_folder: Path, conn: sqlite3.Connection, cfg: BuildCfg
):
    assert image_dir.exists() and index_folder.exists()
    """Builds number of Annoy Indexes from folder of images.
    Returns: list of annoy indexes, NOTE: these are only in memory
    """
    model = cfg.load_model()
    index = cfg.load_index()
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
        image_embeddings = create_image_embeddings(
            model, bimg_path, cfg.image_resolution
        )
        next_id = get_next_table_id(cur, "annoy_index")
        index_path = Path(index_folder) / f"{next_id}.ann"

        index.add_items(image_embeddings)
        index.build(index_path, cfg)
        index.unload()

        insert_indexes_db(cur, index_path, bimg_path)

        time_per_img = (time() - start_time) / len(bimg_path)
        mean_time_per_img = (mean_time_per_img * i + time_per_img) / (i + 1)
        index_pbar.set_description(
            "%d/%d images, %.2fs/img"
            % (index_slice + index_size, nimgs, mean_time_per_img)
        )
    conn.commit()


def insert_indexes_db(cur: sqlite3.Cursor, index_path: Path, bimg_path: list[str]):
    cur.execute("INSERT INTO annoy_index (index_path) VALUES (?)", (str(index_path),))
    index_id = cur.lastrowid

    image_ids = get_image_ids(cur, bimg_path)
    annoy_index_images = [(index_id, j, imid) for j, imid in enumerate(image_ids)]
    cur.executemany(
        "INSERT INTO annoy_index_image (index_id, ref_id, image_id) VALUES (?, ?, ?)",
        annoy_index_images,
    )
