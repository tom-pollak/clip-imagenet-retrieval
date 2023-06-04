# %%
import os
import torch
import numpy as np
from torch import Tensor
from pathlib import Path
import matplotlib.pyplot as plt

import sys

sys.path.append("..")
sys.path.append("../scripts/")

# %%
from clip_index.imagenet import ImagenetDETDataset

image_dir = Path("/Volumes/T7/ILSVRC/Data/DET/val")
image_paths = [image_dir / file for file in os.listdir(image_dir)]
imagenet_root = Path("/Volumes/T7/ILSVRC/")
imagenet_synset = Path("../assets/imagenet_synset_det.txt")

nimgs = len(image_paths)
dataset = ImagenetDETDataset(imagenet_root, imagenet_synset)

correct_classes_imgs = [
    dataset.get_classes_from_image_path(image_paths[i]) for i in range(nimgs)
]
queries = [cls for cls in dataset._synset2desc.values()]

# %%
with torch.no_grad():
    image_embeddings_base_16 = torch.load(
        "../assets/tensors/image_embeddings_base_16.pt"
    )
    query_embeddings_base_16 = torch.load(
        "../assets/tensors/query_embeddings_base_16.pt"
    )
    image_embeddings_base_16 /= image_embeddings_base_16.norm(dim=-1, keepdim=True)
    query_embeddings_base_16 /= query_embeddings_base_16.norm(dim=-1, keepdim=True)
    angular_distances_base_16: Tensor = np.arccos(
        image_embeddings_base_16 @ query_embeddings_base_16.T
    )

# %%
top5_dists, top5_idxs = (angular_distances_base_16).topk(5, largest=False, dim=1)
top1_idxs, top1_dists = top5_idxs[:, 0], top5_dists[:, 0]
top1_classes = [queries[idx] for idx in top1_idxs]
top5_classes = list(map(lambda r: [queries[i] for i in r], top5_idxs))
# %%
count = 0
count2 = 0
for i in range(0, 20120):
    top1_correct = top1_classes[i] in correct_classes_imgs[i]
    top5_correct = any([c in correct_classes_imgs[i] for c in top5_classes[i]])
    cond = 'tennis ball' in top5_classes[i] and 'dog' in top5_classes[i] and 'tennis ball' not in correct_classes_imgs[i] and 'dog' in correct_classes_imgs[i]
    cond2 = 'dog' in top5_classes[i] and 'tennis ball' not in correct_classes_imgs[i] and 'dog' in correct_classes_imgs[i]
    if cond2:
        count2 += 1
    # cond = not top1_correct and top5_correct
    if cond:
        count += 1
        print(top1_correct, top5_correct)
        print(image_paths[i])
        print("correct classes:", ", ".join(correct_classes_imgs[i]))
        for c, d in zip(top5_classes[i], top5_dists[i]):
            print(f"\t{c}: {d:.4f}")
print(count, count2)

# %%
empty = 0
for i in range(20120):
    if not correct_classes_imgs[i]:
        empty += 1
print(empty)

# %%

def show_img(ax, img_path, desc):
    ax.imshow(plt.imread(img_path))
    # ax.set_title(title)
    ax.text(-0.05, 0.95, desc, transform=ax.transAxes, fontsize=8, va="top", ha="right")
    ax.set_xticks([])
    ax.set_yticks([])

# %%

# fig, axes = plt.subplots(1, 3)
# (ax1, ax2, ax3) = axes.flatten()

# show_img(ax1, "/Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00010009.JPEG")
# show_img(ax2, "/Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00048456.JPEG")

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 6))
(ax1, ax2, ax3, ax4) = axes.flatten()

show_img(
    ax1,
    "/Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00032528.JPEG",
    """correct class: fox
fox: 1.3189
dog: 1.3305
antelope: 1.3402
porcupine: 1.3455
domestic cat: 1.3487""",
)


show_img(
    ax2,
    "/Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00008439.JPEG",
    """correct class: lizard
otter: 1.3132
hippopotamus: 1.3210
lizard: 1.3272
seal: 1.3446
armadillo: 1.3490""",
)

# show_img(
#     ax3,
#     "/Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00040247.JPEG",
#     """correct class: lizard
# seal: 1.3098
# hippopotamus: 1.3138
# turtle: 1.3173
# otter: 1.3228
# ray: 1.3357""",
# )

show_img(ax3, "/Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2013_val_00002234.JPEG",
         """correct classes:
         purse    
         backpack    
         person    
coffee maker: 1.3333
plate rack: 1.3586
apple: 1.3731
cream: 1.3733
refrigerator: 1.3734""")

show_img(
    ax4,
    "/Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2013_val_00003188.JPEG",
    """correct class: person
camel: 1.2891
orange: 1.3440
scorpion: 1.3445
coffee maker: 1.3508
cart: 1.3561""",
)
# show_img(ax3, "/Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00040247.JPEG", "incorrect")
# show_img(ax4, "/Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2013_val_00003188.JPEG", "insufficient labels")

fig.savefig("../assets/images/vitb16_summary.png", dpi=300, bbox_inches="tight")

plt.show()
# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 6))
(ax1, ax2, ax3, ax4) = axes.flatten()

show_img(ax1, "/Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00018069.JPEG",
"""correct class: dog
snowplow: 1.3197
snowmobile: 1.3415
balance beam: 1.3447
dog: 1.3452
tennis ball: 1.3453
""")

show_img(ax2, "/Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00000228.JPEG",
"""correct class: dog
snowplow: 1.3312
dog: 1.3362
snowmobile: 1.3421
ski: 1.3439
balance beam: 1.3524
""")

show_img(ax3, "/Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00044944.JPEG",
"""correct classes: dog
dog: 1.3098
tennis ball: 1.3120
croquet ball: 1.3262
puck: 1.3301
banjo: 1.3323""")

show_img(ax4, "/Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2013_val_00002776.JPEG",
"""correct classes: backpack, crutch, person
iPod: 1.3144
crutch: 1.3156
stethoscope: 1.3171
stretcher: 1.3210
horizontal bar: 1.3256""")

fig.savefig("../assets/images/found_images_similar_nlp.png", dpi=300, bbox_inches="tight")

# ['backpack', 'crutch', 'person'] False True
# /Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2013_val_00002776.JPEG

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 6))
(ax1, ax2, ax3, ax4) = axes.flatten()

show_img(ax1, "/Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00040004.JPEG",
"""correct classes: dog
dog: 1.3305
hotdog: 1.3371
tennis ball: 1.3395
squirrel: 1.3409
lion: 1.3438""")

show_img(ax2, "/Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00030756.JPEG",
"""correct classes: dog
dog: 1.3092
puck: 1.3116
banjo: 1.3191
tennis ball: 1.3203
beaker: 1.3262""")

show_img(ax3, "/Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00044944.JPEG",
"""correct classes: dog
dog: 1.3098
tennis ball: 1.3120
croquet ball: 1.3262
puck: 1.3301
banjo: 1.3323""")

show_img(ax4, "/Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00045482.JPEG",
"""correct classes: dog
dog: 1.2901
banjo: 1.2971
fox: 1.2995
squirrel: 1.3073
tennis ball: 1.3110""")
# %%
# ViT-B/16

# Well classified -- we can see that it identifies the more interesting object in the embedding usually
# ['person', 'snowplow']  False True
# /Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00010009.JPEG
# 	snowmobile: 1.2796
# 	snowplow: 1.2901
# 	golfcart: 1.3248
# 	chain saw: 1.3363
# 	stretcher: 1.3473

# can see it gives many brass instruments
# ['trumpet'] False True
# /Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00048456.JPEG
# 	trombone: 1.2509
# 	trumpet: 1.2537
# 	saxophone: 1.2658
# 	french horn: 1.2768
# 	flute: 1.2837

# Top5 correct, ladle top1 wrong
# ['bowl', 'cup or mug'] False True
# /Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2013_val_00001394.JPEG
# 	ladle: 1.3320
# 	cup or mug: 1.3354
# 	bowl: 1.3364
# 	strainer: 1.3414
# 	can opener: 1.3515

# correct classification
# ['fox'] True True
# /Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00032528.JPEG
# 	fox: 1.3189
# 	dog: 1.3305
# 	antelope: 1.3402
# 	porcupine: 1.3455
# 	domestic cat: 1.3487

# ---

# fooled

# hard, because it is in water clip gives many aquatic animals
# ['lizard'] False True
# /Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00008439.JPEG
# 	otter: 1.3132
# 	hippopotamus: 1.3210
# 	lizard: 1.3272
# 	seal: 1.3446
# 	armadillo: 1.3490

# interesting, there is a horizontal bar in the scene, and a plate rack. The model also
# may classify the bench as a stretcher, which is interesting
# ['person', 'dumbbell'] True True
# /Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00004949.JPEG
# 	dumbbell: 1.2790
# 	horizontal bar: 1.2941
# 	plate rack: 1.3283
# 	stretcher: 1.3349
# 	punching bag: 1.3445


# fooled by corkscrew
# ['snake'] False True
# /Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00044475.JPEG
# 	corkscrew: 1.2969
# 	tennis ball: 1.3045
# 	snake: 1.3072
# 	fig: 1.3129
# 	chain saw: 1.3142

# looks like a plastic bag
# ['punching bag'] False True
# /Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00013561.JPEG
# 	plastic bag: 1.2873
# 	punching bag: 1.2956
# 	skunk: 1.3108
# 	purse: 1.3116
# 	wine bottle: 1.3237

# --
# hard label

# interesting label, "watercraft", I'm sure a boat would label better, and a disadvantage
# of zero-shot classification, I'm sure a model trained on this dataset would do better
# ['watercraft'] False True
# /Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00032178.JPEG
# 	oboe: 1.3302
# 	orange: 1.3342
# 	maraca: 1.3500
# 	watercraft: 1.3508
# 	whale: 1.3561

# ---

# incorrect classification

# incorrect classification
# ['lizard'] False False
# /Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00040247.JPEG
# 	seal: 1.3098
# 	hippopotamus: 1.3138
# 	turtle: 1.3173
# 	otter: 1.3228
# 	ray: 1.3357

# completely wrong, does identify the baby bed, but I guess its a dog bed
# ['dog'] False False
# /Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00045634.JPEG
# 	baby bed: 1.3033
# 	pretzel: 1.3154
# 	hotdog: 1.3276
# 	tennis ball: 1.3287
# 	puck: 1.3296

# ['purse', 'backpack', 'person'] False False
# /Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2013_val_00002234.JPEG
# 	coffee maker: 1.3333
# 	plate rack: 1.3586
# 	apple: 1.3731
# 	cream: 1.3733
# 	refrigerator: 1.3734

# ---

# We can see how clip links the "dog" with the "tennis ball" in the scene, and a "snowplow"
# and "snowmobile" from the snow

# ['dog'] False True
# /Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00018069.JPEG
# 	snowplow: 1.3197
# 	snowmobile: 1.3415
# 	balance beam: 1.3447
# 	dog: 1.3452
# 	tennis ball: 1.3453

# ['dog'] False True
# /Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00000228.JPEG
# 	snowplow: 1.3312
# 	dog: 1.3362
# 	snowmobile: 1.3421
# 	ski: 1.3439
# 	balance beam: 1.3524

# ['dog'] False True
# /Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00039515.JPEG
# 	tennis ball: 1.3381
# 	dog: 1.3408
# 	golf ball: 1.3465
# 	croquet ball: 1.3488
# 	oboe: 1.3491

# ['brassiere', 'person'] False True
# /Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2012_val_00005483.JPEG
# 	maillot: 1.3341
# 	whale: 1.3349
# 	brassiere: 1.3403
# 	watercraft: 1.3418
# 	bathing cap: 1.3434

# ---
# I found that a non insignificant number of images were insufficiently labeled, many times
# clip would correctly identify an unlabeled object and get marked incorrect
# ['person', 'table', 'flower pot'] False True
# /Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2013_val_00004196.JPEG
# 	chair: 1.3707
# 	table: 1.3727
# 	flower pot: 1.3823
# 	croquet ball: 1.3838
# 	sofa: 1.3859

# not enough labels, definitely a bookshelf
# ['pencil box'] False True
# /Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2013_val_00001459.JPEG
# 	bookshelf: 1.3190
# 	filing cabinet: 1.3270
# 	plate rack: 1.3319
# 	pencil box: 1.3416
# 	binder: 1.3556

# another incorrect label
# ['person'] False False
# /Volumes/T7/ILSVRC/Data/DET/val/ILSVRC2013_val_00003188.JPEG
# 	camel: 1.2891
# 	orange: 1.3440
# 	scorpion: 1.3445
# 	coffee maker: 1.3508
# 	cart: 1.3561
# ---
