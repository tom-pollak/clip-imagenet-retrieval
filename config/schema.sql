PRAGMA foreign_keys = OFF;
BEGIN TRANSACTION;
CREATE TABLE image (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT UNIQUE NOT NULL
);
CREATE TABLE annoy_index (
    index_id INTEGER PRIMARY KEY AUTOINCREMENT,
    index_path TEXT UNIQUE NOT NULL
);
CREATE TABLE annoy_index_image (
    index_id INTEGER NOT NULL,
    image_id INTEGER NOT NULL,
    ref_id INTEGER NOT NULL,
    PRIMARY KEY (index_id, image_id),
    FOREIGN KEY(index_id) REFERENCES annoy_index(index_id),
    FOREIGN KEY(image_id) REFERENCES image(image_id)
);
COMMIT;
