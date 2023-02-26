# %%
import sys

sys.path.append("..")

import math
import os
import pickle
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from annoy import AnnoyIndex

# %%
image_dir = Path("/Volumes/T7/ILSVRC/Data/DET/val")
image_paths = [image_dir / file for file in os.listdir(image_dir)]

# %%
from clip_index.imagenet import ImagenetDETDataset

imagenet_root = Path("/Volumes/T7/ILSVRC/")
imagenet_synset = Path("../assets/imagenet_synset_det.txt")

dataset = ImagenetDETDataset(imagenet_root, imagenet_synset)

queries = [cls for cls in dataset._synset2desc.values()]


# %%
image_embeddings = torch.load("../assets/tensors/image_embeddings.pt")
query_embeddings = torch.load("../assets/tensors/query_embeddings.pt")

print(image_embeddings.shape, query_embeddings.shape)

# %%
# Pick the top 5 most similar labels for the image
image_embeddings /= torch.linalg.norm(image_embeddings, dim=1, keepdim=True)
query_embeddings /= torch.linalg.norm(query_embeddings, dim=1, keepdim=True)
similarity = F.softmax(100.0 * image_embeddings @ query_embeddings.T, dim=1)
# similarity[0].topk(1)

# %%
# Visualize the top 5 most similar labels for the image
# Crashes alot
# print("Top 1 predictions:")
# fig, ax = plt.subplots(5, 2, figsize=(6, 12))
# for i in range(10):
#     value, index = similarity[i].topk(1)
#     correct_classes = dataset.get_classes_from_image_path(image_paths[i])
#     predicted_class = queries[index.item()]

#     img = plt.imread(image_paths[i])
#     img_plt = ax[i % 5][i % 2]
#     img_plt.imshow(img)
#     img_plt.set_title(
#         f"{correct_classes}, {predicted_class} -- {value.item():.3f}", fontsize=8
#     )
#     img_plt.set_axis_off()
# plt.show()

# %%
# TOP 1 and TOP 5 accuracy
nimgs = len(image_paths)
top_1_correct = 0
top_5_correct = 0
for i in range(nimgs):
    correct_classes = dataset.get_classes_from_image_path(image_paths[i])

    # TOP 1
    value, index = similarity[i].topk(1)
    predicted_class = queries[index.item()]
    confidence = value.item()
    if predicted_class in correct_classes:
        top_1_correct += 1

    # TOP 5
    value, index = similarity[i].topk(5)
    predicted_classes = [queries[idx.item()] for idx in index]
    if any([cls in correct_classes for cls in predicted_classes]):
        top_5_correct += 1

print(f"Top 1 accuracy: {top_1_correct / nimgs:.3f}")
print(f"Top 5 accuracy: {top_5_correct / nimgs:.3f}")

# %%
# Build indexes
index_sizes = [16, 64, 128, 512, 1024, 4096, 8192, 20120]
ntrees = [64, 256, 1024, 4096, 16384]

bstats = {"Annoy": {"index_size": {}, "ntrees": {}}, "L2": {}}


def build_indexes():
    from clip_index.image.build import build_indexes_from_image_folder
    from clip_index.annoy import AnnoyBuildCfg
    import sqlite3

    conn = sqlite3.connect("/Users/tom/projects/clip-index/assets/db/clip_image.db")
    for isize in index_sizes:
        print(f"Building index with index_size={isize}")
        cfg = AnnoyBuildCfg(index_size=isize, ntrees=None)
        bstat = build_indexes_from_image_folder(
            image_dir=Path("/Volumes/T7/ILSVRC/Data/DET/val/"),
            index_folder=Path(
                f"/Volumes/T7/clip-indexes/motis/annoy/index_sizes/{isize}/"
            ),
            conn=conn,
            cfg=cfg,
            cache_image_embeddings=image_embeddings,
        )

        bstats["Annoy"]["index_size"][isize] = bstat
    print()
    for nt in ntrees:
        print(f"Building index with ntrees={nt}")
        cfg = AnnoyBuildCfg(index_size=512, ntrees=nt)
        bstat = build_indexes_from_image_folder(
            image_dir=Path("/Volumes/T7/ILSVRC/Data/DET/val/"),
            index_folder=Path(f"/Volumes/T7/clip-indexes/motis/annoy/ntrees/{nt}/"),
            conn=conn,
            cfg=cfg,
            cache_image_embeddings=image_embeddings,
        )
        bstats["Annoy"]["ntrees"][nt] = bstat

    pickle.dump(bstats, open("../assets/pickles/build_stats.pkl", "wb"))


# build_indexes()

# %%
import os
import sys

sys.path.append(os.path.abspath(__file__))
from query_res import (
    ImagewiseResult,
    QuerywiseResult,
    evaluate_queries,
    result_to_dict,
)

from clip_index.annoy import AnnoyQueryCfg

max_results = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
index_size_index_folders = [
    Path(f"/Volumes/T7/clip-indexes/motis/annoy/index_sizes/{isize}/")
    for isize in index_sizes
]

ntrees_index_folders = [
    Path(f"/Volumes/T7/clip-indexes/motis/annoy/ntrees/{nt}/") for nt in ntrees
]


# %%
print("Imagewise max results vs index size")
max_results_res: dict[int, dict] = {}
for max_result in max_results:
    res = evaluate_queries(
        ImagewiseResult,
        index_size_index_folders,
        AnnoyQueryCfg(max_results_per_query=max_result),
    )
    res_dict = result_to_dict(res)
    max_results_res[max_result] = res_dict

max_results_path = Path("../assets/pickles/imagewise_max_results_index_size.pkl")
pickle.dump(max_results_res, open(max_results_path, "wb"))
print(f"Saved to {max_results_path}\n")

# %%
print("Querywise max results=5 vs index size")
res = evaluate_queries(
    QuerywiseResult, index_size_index_folders, AnnoyQueryCfg(max_results_per_query=5)
)
res_dict = result_to_dict(res)

index_size_path = Path("../assets/pickles/querywise_top5_index_size.pkl")
pickle.dump(res_dict, open(index_size_path, "wb"))
print(f"Saved to {index_size_path}")

# %%
print("Querywise max results=5 vs ntrees")
res = evaluate_queries(
    QuerywiseResult, ntrees_index_folders, AnnoyQueryCfg(max_results_per_query=5)
)
res_dict = result_to_dict(res)

ntrees_path = Path("../assets/pickles/querywise_top5_ntrees.pkl")
pickle.dump(res_dict, open(ntrees_path, "wb"))
print(f"Saved to {ntrees_path}")

# %%
