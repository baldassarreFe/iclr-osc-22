"""
K-means implementations.
"""
import numpy as np
import torch
from torch import Tensor


@torch.no_grad()
def torch_kmeans_euclidean(
    x: Tensor, num_clusters: int, seed: int = None, max_iters=10, tol=1e-8
):
    """Batched k-means for PyTorch with Euclidean distance.

    Args:
        x: tensor containing ``B`` batches of ``N`` ``C``-dimensional
            tensors each, shape [B N C]
        num_clusters: number of clusters ``K``
        seed: int seed for centroid initialization, leave empty to use
            numpy default generator
        max_iters: max number of iterations
        tol: tolerance for early termination

    Returns:
        A ``[B K C]`` tensor of centroids.
    """
    B, N, C = x.shape
    K = num_clusters

    # [B K] indexes of the initial centroids
    rng = np.random if seed is None else np.random.default_rng(seed)
    idx = rng.choice(N, size=K, replace=False)
    idx = torch.from_numpy(idx).expand(B, -1).to(x.device)

    # [B K C] gather centroids from x using the sampled indexes
    centroids = x.gather(dim=1, index=idx[:, :, None].expand(-1, -1, C))

    for i in range(max_iters):
        # [B N K] mean squared error between each point and each centroid
        mse = (centroids[:, None, :, :] - x[:, :, None, :]).pow_(2).sum(dim=-1)

        # [B N] assign the closest centroid to each sample
        idx = mse.argmin(dim=-1)

        # [B K] how many samples are assigned to each centroid?
        counts = idx.new_zeros(B, K).scatter_add_(
            dim=1, index=idx, src=idx.new_ones(B, N)
        )

        # [B K C] for each centroid, sum all samples assigned to it, divide by count.
        # If centroid has zero point assigned to it, leave the old centroid value
        new_centr = (
            torch.zeros_like(centroids)
            .scatter_add_(dim=1, index=idx[:, :, None].expand(-1, -1, C), src=x)
            .div_(counts[:, :, None])
        )
        new_centr = torch.where(counts[:, :, None] == 0, centroids, new_centr)

        # Compute tolerance and exit early
        error = centroids.sub_(new_centr).abs_().max()
        centroids = new_centr
        if error.item() < tol:
            break

    return centroids


@torch.no_grad()
def torch_kmeans_cosine(
    x: Tensor, num_clusters: int, seed: int = None, max_iters=10, tol=1e-8
):
    """Batched k-means for PyTorch with cosine distance.

    Args:
        x: tensor containing ``B`` batches of ``N`` ``C``-dimensional
            tensors each, shape [B N C]. The tensors should be L2-normalized
            along the ``C`` dimension.
        num_clusters: number of clusters ``K``
        seed: int seed for centroid initialization, leave empty to use
            numpy default generator
        max_iters: max number of iterations
        tol: tolerance for early termination

    Returns:
        A ``[B K C]`` tensor of centroids.
    """
    B, N, C = x.shape
    K = num_clusters

    # [B K] indexes of the initial centroids
    rng = np.random if seed is None else np.random.default_rng(seed)
    idx = torch.from_numpy(rng.choice(N, size=(B, K), replace=True)).to(x.device)

    # [B K C] gather centroids from x using the sampled indexes
    centroids = x.detach().gather(dim=1, index=idx[:, :, None].expand(-1, -1, C))

    for i in range(max_iters):
        # [B N K] mean squared error between each point and each centroid.
        cos = torch.einsum("bnc, bkc -> bnk", x, centroids)

        # [B N] assign the closest centroid to each sample
        idx = cos.argmax(dim=-1)

        # [B K] how many samples are assigned to each centroid?
        counts = (
            idx.new_zeros(B, K)
            .scatter_add_(dim=1, index=idx, src=idx.new_ones(B, N))
            .clamp_min(1)
        )

        # [B K C] for each centroid, sum all samples assigned to it,
        # divide by count and re-normalize
        new_centr = torch.zeros_like(centroids)
        new_centr.scatter_add_(dim=1, index=idx[:, :, None].expand(-1, -1, C), src=x)
        new_centr.div_(counts[:, :, None])
        new_centr.div_(torch.linalg.vector_norm(new_centr, dim=-1, keepdim=True))

        # Compute tolerance and exit early
        error = centroids.sub_(new_centr).abs_().max()
        centroids = new_centr
        if error.item() < tol:
            break

    return centroids
