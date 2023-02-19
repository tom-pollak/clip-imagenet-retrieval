import argparse
import json
import sqlite3
from pathlib import Path
from time import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Query CLIP")
    parser.add_argument(
        "-c",
        "--config",
        help="Json config for CLIP Index",
    )
    args = parser.parse_args()
    return args


def run_build(config: dict):
    from clip_index.image.build import build_indexes_from_imgs
    from clip_index.utils.config import BuildCfg

    print("Starting index build")
    assert config.get("image_dir", False), "Please give image_dir in config"
    assert config.get("index_folder", False), "Please give index_folder in config"
    conn = sqlite3.connect(config["sqlite_path"])
    start_time = time()
    cfg = BuildCfg.from_dict(config)
    model = cfg.load_model()
    build_indexes_from_imgs(
        model=model,
        image_dir=Path(config["image_dir"]),
        index_folder=Path(config["index_folder"]),
        conn=conn,
        cfg=cfg,
    )
    conn.close()
    print("\nBuild finished in: ", round(time() - start_time, 3))


def run_query(config: dict):
    from clip_index.text.query import filter_closest_n_results, query_index
    from clip_index.utils.config import QueryCfg
    from clip_index.utils.demo import demo_images

    print("Querying index folder...")
    start_time = time()
    assert config.get("index_folder", False), "Please give index_folder in config"
    assert config.get("queries", False), "Please give queries in config"
    conn = sqlite3.connect(config["sqlite_path"])
    cur = conn.cursor()
    cfg = QueryCfg.from_dict(config)
    annoy_queries = query_index(
        queries=config["queries"],
        index_folder=Path(config["index_folder"]),
        cur=cur,
        cfg=cfg,
    )
    conn.close()

    print(
        f"Found {sum([len(i) for i in annoy_queries.values()])} images in {time() - start_time:.2f} seconds"
    )

    max_demo_results = config.get("max_results_demo", None)
    if max_demo_results is not None:
        filter_closest_n_results(annoy_queries, max_demo_results)

    demo_images(annoy_queries)


if __name__ == "__main__":
    args = parse_args()
    assert args.config, "Please give json config with -c"
    with open(args.config) as f:
        config = json.load(f)

    mode = config["mode"]
    if mode == "build":
        run_build(config)
    elif mode == "query":
        run_query(config)
    else:
        raise ValueError(f"mode {mode} does not exist")
