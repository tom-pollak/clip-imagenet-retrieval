# %%
import os
import sys
import time
from pathlib import Path

sys.path.append("..")
import torch

### Create Image Embeddings ###
# %%
image_dir = Path("/Volumes/T7/ILSVRC/Data/DET/val")
img_paths = [str(image_dir / file) for file in os.listdir(image_dir)]
# %%
from clip_index.annoy import AnnoyBuildCfg

config = {"model_name": "clip:RN50x4"}
image_embeddings_path = Path(
    "/Users/tom/projects/clip-index/assets/tensors/image_embeddings_rn50x4.pt"
)
query_embeddings_path = Path(
    "/Users/tom/projects/clip-index/assets/tensors/query_embeddings_rn50x4.pt"
)
embedding_size = 640
device = "cpu"
cfg = AnnoyBuildCfg.from_dict(config)
model = cfg.load_model(device)

# %%

from clip_index.image.build import create_image_embeddings

start_time = time.perf_counter()
image_embeddings, embedding_time = create_image_embeddings(
    model, img_paths, embedding_size=embedding_size, device=device, input_resolution=288
)
print(f"Time to create image embeddings: {embedding_time} seconds")

# %%
torch.save(image_embeddings, image_embeddings_path)

### Create Query Embeddings ###
# %%
from clip_index.imagenet import ImagenetDETDataset

imagenet_root = Path("/Volumes/T7/ILSVRC/")
imagenet_synset = Path("/Users/tom/projects/clip-index/assets/imagenet_synset_det.txt")

dataset = ImagenetDETDataset(imagenet_root, imagenet_synset)

queries = [cls for cls in dataset._synset2desc.values()]

# %%
from clip_index.annoy import AnnoyQueryCfg
from clip_index.text.query import create_query_embeddings

cfg = AnnoyQueryCfg.from_dict(config)
model = cfg.load_model(device)

query_embeddings = create_query_embeddings(model, queries, device=device)

# %%
torch.save(query_embeddings, query_embeddings_path)

# %%
