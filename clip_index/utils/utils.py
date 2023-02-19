import os
from pathlib import Path
from typing import Literal

from annoy import AnnoyIndex

ROOTDIR = Path(os.path.dirname(os.path.abspath(__file__))).parent


def create_index(
    vector_size: int = 512,
    metric: Literal["angular", "euclidean", "manhattan", "hamming", "dot"] = "angular",
) -> AnnoyIndex:
    """
    Parameters:
    - vector_size: size of input, referred as "f" in AnnoyIndex
        - Default 512, this is the size of the clip encoding output
    - metric: metric to find nearest neighbors
    """
    return AnnoyIndex(vector_size, metric)
