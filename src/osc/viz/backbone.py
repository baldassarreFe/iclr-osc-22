"""
Clustering backbone features.
"""
from typing import Tuple, Union

import einops
import numpy as np
import torch
from IPython.core.display import Image
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def kmeans_backbone(images, f_backbone, num_patches: Tuple[int, int]):
    assert f_backbone.shape[1] == np.prod(num_patches)
    num_augs = images.shape[0]
    batch_size = images.shape[1]

    f_backbone = einops.rearrange(
        f_backbone.detach().cpu().numpy(),
        "(A B) P_hw C -> B (A P_hw) C",
        A=num_augs,
        B=batch_size,
    )
    images = einops.rearrange(
        images.detach().cpu().numpy(),
        "A B C H W -> B A H W C",
    )

    for b in range(batch_size):
        fig, axs = plt.subplots(2, num_augs, figsize=np.array(num_patches[::-1]) / 2)

        kmeans = KMeans(init="k-means++", n_clusters=11, n_init=4, random_state=0)
        clust = kmeans.fit_predict(f_backbone[b]).reshape(num_augs, *num_patches)
        for a in range(num_augs):
            axs[0, a].imshow(images[b, a])
            axs[0, a].set_title(f"Aug {a}")
            axs[1, a].imshow(clust[a], cmap="tab20")

        for ax in axs[:-1, :].flat:
            ax.set_xticks([])
        for ax in axs[:, 1:].flat:
            ax.set_yticks([])

        fig.suptitle(f"Image {b} - backbone features K-Means")
        fig.tight_layout()
        fig.set_facecolor("white")
        fig.savefig(f"img-{b}-vit-kmeans.png", dpi=200)
        plt.close(fig)
        display(Image(url=f"img-{b}-vit-kmeans.png", width=600))


def kmeans_clusters(
    x: Union[torch.Tensor, np.ndarray], n_clusters: int = 11
) -> np.ndarray:
    """Batched K-means clustering.

    Args:
        x: N samples of C-dimensional features with leading batch dimensions,
           e.g. shape [..., N, C]
        n_clusters: desired number of clusters

    Returns:
        int array of cluster IDs, shape [..., N, C]
    """

    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    res = np.empty(x.shape[:-1], dtype=int)
    for b in np.ndindex(x.shape[:-2]):
        kmeans = KMeans(
            init="k-means++", n_clusters=n_clusters, n_init=4, random_state=0
        )
        res[b] = kmeans.fit_predict(x[b])

    return res
