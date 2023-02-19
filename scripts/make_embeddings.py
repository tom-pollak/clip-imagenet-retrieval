#%%
import os
import sys
from pathlib import Path
from time import time

sys.path.append("..")
import torch

### Create Image Embeddings ###
#%%
image_dir = Path("/Volumes/T7/ILSVRC/Data/DET/val")
img_paths = [str(image_dir / file) for file in os.listdir(image_dir)]

#%%
from clip_index.utils.config import BuildCfg

config = {"model_name": "motis"}
cfg = BuildCfg.from_dict(config)
model = cfg.load_model()


#%%
from clip_index import create_image_embeddings

start_time = time()
image_embeddings = create_image_embeddings(model, img_paths)
print(f"Time to create image embeddings: {time() - start_time:.3f} seconds")

# %%
torch.save(image_embeddings, "../assets/image_embeddings.pt")

### Create Query Embeddings ###
#%%
from clip_index.imagenet import ImagenetDETDataset

imagenet_root = Path("/Volumes/T7/ILSVRC/")
imagenet_synset = Path("../assets/imagenet_synset_det.txt")

dataset = ImagenetDETDataset(imagenet_root, imagenet_synset)

queries = [cls for cls in dataset._synset2desc.values()]

#%%
from clip_index.text.query import create_query_embeddings
from clip_index.utils.config import QueryCfg

cfg = QueryCfg()
model = cfg.load_model()

query_embeddings = create_query_embeddings(model, queries)

#%%
torch.save(query_embeddings, "../assets/query_embeddings.pt")
