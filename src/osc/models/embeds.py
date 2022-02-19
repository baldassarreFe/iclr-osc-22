"""
Positional embeddings and query tokens for objects.
"""

import numpy as np
import timm.models
import torch
from torch import Tensor
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
    """Dynamically sample ``S`` object tokens from ``K`` gaussian components.

    See :func:``forward``.
    """

    def __init__(self, embed_dim: int, num_objects: int, num_components: int = 1):
        """

        Args:
            embed_dim: embedding dimension ``D``
            num_objects: default number of objects tokens to sample ``S``
            num_components: number of gaussian components to learn ``K``.
                If greater than one, each object token is drawn from one of
                ``K`` gaussians chosen uniformly at random with replacement.
        """
        super(SampledObjectTokens, self).__init__()
        self.mu = nn.Parameter(torch.zeros(num_components, embed_dim))
        self.log_sigma = nn.Parameter(torch.zeros(num_components, embed_dim))
        self.init_weights()
        self.num_objects = num_objects

    def init_weights(self):
        """Glorot/Xavier uniform initialization"""
        embed_dim = self.mu.shape[-1]
        limit = np.sqrt(6.0 / embed_dim)
        nn.init.uniform_(self.mu, -limit, +limit)
        nn.init.uniform_(self.log_sigma, -limit, +limit)

    def forward(self, batch_size: int, num_objects: int = None, seed: int = None):
        """Forward.

        Args:
            batch_size: number of independent batches ``B``
            num_objects: number of object tokens to sample ``S``
            seed: optional random seed

        Returns:
            A tensor of sampled object tokens with shape ``[B S D]``
        """
        device = self.mu.device
        B = batch_size
        S = num_objects if num_objects is not None else self.num_objects
        K, D = self.mu.shape

        # Pick which random generator to use
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
        #       but that makes val loss artificially low so the training
        #       behavior is always enabled by doing `if True or` in two places

        # Sample from from a normal gaussian
        if True or self.training:
            # Sample B*S independent vectors of length D
            slots = torch.randn(B, S, D, device=device, generator=rng)
        else:
            # Sample S independent vectors of length D, then reuse them for all B images
            slots = torch.randn(1, S, D, device=device, generator=rng).expand(B, -1, -1)

        # If the parameters describe a single component, use that single mu and sigma.
        # Otherwise uniformly sample S (mu, sigma) pairs out of the K learned pairs.
        if K == 1:
            mu = self.mu
            log_sigma = self.log_sigma
        else:
            if True or self.training:
                # Choose S components for each image in the batch independently
                idx = torch.randint(0, K, size=(B * S,), device=device, generator=rng)
            else:
                # Choose the same S components for all images in the batch
                idx = torch.randint(0, K, size=(S,), device=device, generator=rng)
            mu = self.mu[idx, :].reshape(-1, S, D).expand(B, S, D)
            log_sigma = self.log_sigma[idx, :].reshape(-1, S, D).expand(B, S, D)

        # Combine learned parameters (mu, sigma) and normal-distributed samples
        return mu + log_sigma.exp() * slots


class KmeansCosineObjectTokens(nn.Module):
    def __init__(self, num_objects: int):
        super(KmeansCosineObjectTokens, self).__init__()
        self.max_iters = 20
        self.tol = 1e-6
        self.num_objects = num_objects

    def forward(self, x: Tensor, num_objects: int = None, seed=None):
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

    def forward(self, x: Tensor, num_objects: int = None, seed=None):
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
