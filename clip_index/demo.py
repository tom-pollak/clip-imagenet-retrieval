# pyright: reportGeneralTypeIssues=false

import math

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
from clip_index.annoy import AnnoyQueries
from clip_index.index import AnnQueries


def demo_images(query_image_dict: AnnQueries | AnnoyQueries):
    for query, images in query_image_dict.items():
        for im in images:
            assert (
                im.image_path
            ), "Please add image path to AnnoyImage (see clip_index/text/query.py:add_image_path)"
            assert im.image_path.exists(), f"Image path does not exist: {im.image_path}"

            img = cv2.imread(str(im.image_path))
            cv2.imshow(f"{query}: {im.dist}", img)
            try:
                cv2.waitKey(0)
            # Exit gracefully by <C-c> then clicking to next image
            except KeyboardInterrupt:
                cv2.destroyAllWindows()
    cv2.destroyAllWindows()


def show_queries(annoy_qs: AnnoyQueries) -> Figure:
    fig = plt.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(len(annoy_qs), 1)
    for i, (query, images) in enumerate(annoy_qs.items()):
        row_title = fig.add_subplot(outer[i])
        row_title.axis("off")
        row_title.set_title(query + "\n")

        num_imgs = len(list(annoy_qs.values())[i])
        rows = num_imgs // 4
        cols = math.ceil(num_imgs / 4 * 3)
        if rows * cols < num_imgs:
            rows += 1  # math.ceil

        inner = gridspec.GridSpecFromSubplotSpec(
            rows, cols, subplot_spec=outer[i], wspace=0.1, hspace=0.1
        )
        for j, im in enumerate(images):
            assert (
                im.image_path
            ), "Please add image path to AnnoyImage (see clip_index/text/query.py:add_image_path)"
            img = Image.open(im.image_path)
            ax = plt.Subplot(fig, inner[j])
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"{im.dist:.4f}", fontsize=9)
            fig.add_subplot(ax)
    return fig
