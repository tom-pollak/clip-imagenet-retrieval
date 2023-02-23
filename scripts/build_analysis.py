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
image_embeddings = torch.load("../assets/image_embeddings.pt")
query_embeddings = torch.load("../assets/query_embeddings.pt")

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
nimgs = len(image_paths)


def build_indexes():
    from clip_index.image.build_index import add_images_to_index

    print(f"Building indexes...")
    for isize in index_sizes:
        nindexes = math.ceil(nimgs / isize)
        tot_index_add_time = 0
        tot_build_time = 0
        cpu_build_time = 0
        for i in range(nindexes):
            index = AnnoyIndex(512, "angular")
            index.on_disk_build(f"/Volumes/T7/clip-indexes/motis/val/{isize}/{i}.ann")

            start_time = time.perf_counter()
            add_images_to_index(index, image_embeddings[i * isize : (i + 1) * isize])
            finish_add_items = time.perf_counter()
            cpu_time_start_build = time.process_time()
            index.build(index.f * 2)
            finish_build = time.perf_counter()
            cpu_time_finish = time.process_time()

            tot_index_add_time += finish_add_items - start_time
            tot_build_time += finish_build - finish_add_items
            cpu_build_time += cpu_time_finish - cpu_time_start_build

            index.unload()

        log = f"""Built indexes of size {isize} ({nindexes} indexes)
        Index add time: {tot_index_add_time:.5f}, {tot_index_add_time / nindexes:.5f} per index
        Build time: {tot_build_time:.5f}, {tot_build_time / nindexes:.5f} per index
        CPU build time: {cpu_build_time:.5f}, {cpu_build_time / nindexes:.5f} per index\n"""
        print(log)
        with open(
            f"/Volumes/T7/clip-indexes/motis/val/{isize}/build_log.txt", "w"
        ) as f:
            f.write(log)


# build_indexes() # You probably don't want to run this

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

index_sizes = [16, 64, 128, 512, 1024, 4096, 8192, 20120]
max_results = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
index_folders = [
    Path(f"/Volumes/T7/clip-indexes/motis/val/{isize}/") for isize in index_sizes
]


# %%
print("Creating imagewise results...")
max_results_res: dict[int, dict] = {}
for max_result in max_results:
    res = evaluate_queries(
        ImagewiseResult, index_folders, AnnoyQueryCfg(max_results_per_query=max_result)
    )
    res_dict = result_to_dict(res)
    max_results_res[max_result] = res_dict

max_results_path = Path("../assets/imagewise_ann_queries_max_results_all_indexes.pkl")
pickle.dump(max_results_res, open(max_results_path, "wb"))
print(f"Saved to {max_results_path}\n")

# %%
print("Creating querywise results...")
res = evaluate_queries(
    QuerywiseResult, index_folders, AnnoyQueryCfg(max_results_per_query=5)
)
res_dict = result_to_dict(res)

querywise_path = Path("../assets/querywise_ann_queries_top5.pkl")
pickle.dump(res_dict, open(querywise_path, "wb"))
print(f"Saved to {querywise_path}")

# %%
