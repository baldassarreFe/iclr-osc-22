"""
Positional embeddings and query tokens for objects.
"""

import numpy as np
import timm.models
import torch
from torch import nn as nn

from osc.utils import l2_normalize


class PositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embed_dim, dropout=0.0):
        super(PositionalEmbedding, self).__init__()
        self.embed = nn.Parameter(torch.zeros((num_embeddings, embed_dim)))
        self.drop = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        timm.models.layers.trunc_normal_(self.embed, std=0.02)

    def forward(self, x):
        return self.drop(x + self.embed)


class LearnedObjectTokens(nn.Module):
    """Fixed number of learnable object tokens, shape ``[S C]``"""

    def __init__(self, embed_dim: int, num_objects: int):
        super(LearnedObjectTokens, self).__init__()
        self.tokens = nn.Parameter(torch.zeros((num_objects, embed_dim)))
        self.init_weights()

    def init_weights(self):
        timm.models.layers.trunc_normal_(self.tokens, std=0.02)

    def forward(self):
        return self.tokens


class SampledObjectTokens(nn.Module):
    """Dynamically sampled object tokens, shape ``[S C]``"""

    def __init__(self, embed_dim: int, num_objects: int):
        super(SampledObjectTokens, self).__init__()
        self.mu = nn.Parameter(torch.zeros(embed_dim))
        self.log_sigma = nn.Parameter(torch.zeros(embed_dim))
        self.init_weights()
        self.num_objects = num_objects

    def init_weights(self):
        # glorot/xavier uniform initialization
        limit = np.sqrt(6 / self.mu.numel())
        nn.init.uniform_(self.mu, -limit, +limit)
        nn.init.uniform_(self.log_sigma, -limit, +limit)

    def forward(self, batch_size: int, num_objects: int = None, seed: int = None):
        device = self.mu.device
        B = batch_size
        K = num_objects if num_objects is not None else self.num_objects
        C = self.mu.shape[-1]

        if seed is not None:
            rng = torch.Generator(device).manual_seed(seed)
        else:
            if self.training:
                # Use global rng
                rng = None
            else:
                # Use fixed rng
                rng = torch.Generator(device).manual_seed(18327415066407224732)

        # TODO: during val slots should be the same for all images,
        #       but that makes val loss artificially low
        if True or self.training:
            # Sample BK independent vectors of length C
            slots = torch.randn(B, K, C, device=device, generator=rng)
        else:
            # Sample K independent vectors of length C, then reuse them for all B images
            slots = torch.randn(K, C, device=device, generator=rng)
            slots = slots.expand(B, -1, -1)

        slots = self.mu + self.log_sigma.exp() * slots
        return slots


class KmeansCosineObjectTokens(nn.Module):
    def __init__(self, num_objects: int):
        super(KmeansCosineObjectTokens, self).__init__()
        self.max_iters = 20
        self.tol = 1e-6
        self.num_objects = num_objects

    def forward(self, x: torch.Tensor, num_objects: int = None, seed=None):
        """K-means clustering to initialize object tokens.

        Args:
            x: feature tensor, shape``[B N C]``. It will be L2-normalized internally.
            num_objects: number of objects
            seed: int seed, leave empty to use numpy default generator

        Returns:
            A ``[B K C]`` tensor of centroid features.
        """
        if num_objects is None:
            num_objects = self.num_objects
        x = l2_normalize(x)
        return torch_kmeans_cosine(x, num_objects, seed, self.max_iters, self.tol)


class KmeansEuclideanObjectTokens(nn.Module):
    def __init__(self, num_objects: int):
        super(KmeansEuclideanObjectTokens, self).__init__()
        self.max_iters = 20
        self.tol = 1e-6
        self.num_objects = num_objects

    def forward(self, x: torch.Tensor, num_objects: int = None, seed=None):
        """K-means clustering to initialize object tokens.

        Args:
            x: feature tensor, shape``[B N C]``
            num_objects: number of objects
            seed: int seed, leave empty to use numpy default generator

        Returns:
            A ``[B K C]`` tensor of centroid features.
        """
        if num_objects is None:
            num_objects = self.num_objects
        return torch_kmeans_euclidean(x, num_objects, seed, self.max_iters, self.tol)


@torch.no_grad()
def torch_kmeans_euclidean(
    x: torch.Tensor, num_clusters: int, seed: int = None, max_iters=10, tol=1e-8
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
    x: torch.Tensor, num_clusters: int, seed: int = None, max_iters=10, tol=1e-8
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
