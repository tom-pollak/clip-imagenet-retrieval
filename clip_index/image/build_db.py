import sqlite3
from pathlib import Path

from annoy import AnnoyIndex


def insert_images_db(cur: sqlite3.Cursor, image_paths: list[str]):
    """
    Inserts images into sqlite database
    """
    assert all(
        map(lambda path: Path(path).exists(), image_paths)
    ), f"not all paths in {image_paths} exist"
    cur.executemany(
        "INSERT OR IGNORE INTO image (image_path) VALUES (?) ", map(tuple, image_paths)
    )


def get_image_ids(cur: sqlite3.Cursor, image_paths: list[str]) -> list[int]:
    # Not sure if the results returned in order
    # query = "SELECT image_id from image WHERE image_path IN %s" % str(tuple(image_paths))
    ids = []
    for im_path in image_paths:
        img_id = cur.execute(
            "SELECT image_id FROM image WHERE image_path = ?", (im_path,)
        ).fetchone()[0]
        ids.append(img_id)
    assert all(ids), "Some ids are none"
    return ids


def get_next_table_id(cur: sqlite3.Cursor, table: str) -> int:
    next_id = cur.execute(
        f"SELECT * FROM SQLITE_SEQUENCE WHERE name='{table}'"
    ).fetchone()
    if next_id is None:
        next_id = 1
    else:
        next_id = next_id[1] + 1
    return next_id


def insert_indexes_db(
    indexes: list[AnnoyIndex], cur: sqlite3.Cursor, index_folder: Path
) -> list[Path]:
    """
    Saves list of indexes in sqlite database
    """
    next_id = get_next_table_id(cur, "annoy_index")
    index_paths = [
        Path(index_folder) / f"{next_id + i}.ann" for i in range(len(indexes))
    ]
    assert not any(
        map(lambda path: path.exists(), index_paths)
    ), f"Index path already exists, start={next_id}, end={next_id + len(indexes)}"

    for index, path in zip(indexes, index_paths):
        index.save(str(path))
        index.unload()

    cur.executemany(
        """
        INSERT INTO annoy_index (index_path)
        VALUES (?)
        """,
        map(lambda index: (index,), str(index_paths)),
    )
    return index_paths
