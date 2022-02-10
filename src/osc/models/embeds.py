import numpy as np
import timm.models
import torch
from torch import nn as nn


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

    def __init__(self, num_objects, embed_dim):
        super(LearnedObjectTokens, self).__init__()
        self.tokens = nn.Parameter(torch.zeros((num_objects, embed_dim)))
        self.init_weights()

    def init_weights(self):
        timm.models.layers.trunc_normal_(self.tokens, std=0.02)

    def forward(self):
        return self.tokens


class SampledObjectTokens(nn.Module):
    """Dynamically sampled object tokens, shape ``[S C]``"""

    def __init__(self, embed_dim):
        super(SampledObjectTokens, self).__init__()
        self.mu = nn.Parameter(torch.zeros(embed_dim))
        self.log_sigma = nn.Parameter(torch.zeros(embed_dim))
        self.init_weights()

    def init_weights(self):
        # glorot/xavier uniform initialization
        limit = np.sqrt(6 / self.mu.numel())
        nn.init.uniform_(self.mu, -limit, +limit)
        nn.init.uniform_(self.log_sigma, -limit, +limit)

    def forward(self, batch_size: int, num_objects: int, seed: int = None):
        device = self.mu.device
        B = batch_size
        K = num_objects
        C = self.mu.shape[-1]

        if self.training:
            # Use global rng unless seed is explicitly given
            rng = None
            if seed is not None:
                rng = torch.Generator(device).manual_seed(seed)

            # Sample BK independent vectors of length C
            slots = torch.randn(B, K, C, device=device, generator=rng)
            slots = self.mu + self.log_sigma.exp() * slots

        else:
            # Use a one-time rng with fixed seed unless seed is explicitly given
            if seed is None:
                seed = 18327415066407224732
            rng = torch.Generator(device).manual_seed(seed)

            # Sample K independent vectors of length C, then reuse them for all B images
            slots = torch.randn(K, C, device=device, generator=rng)
            slots = self.mu + self.log_sigma.exp() * slots
            slots = slots.expand(B, -1, -1)

        return slots


class KmeansObjectTokens(nn.Module):
    def __init__(self):
        super(KmeansObjectTokens, self).__init__()
        self.max_iters = 20
        self.tol = 1e-6

    def forward(self, f_backbone: torch.Tensor, num_objects: int, seed=None):
        """K-means clustering to initialize object tokens.

        Args:
            f_backbone: feature tensor, shape``[B N C]``
            num_objects: number of objects
            seed: int seed, leave empty to use numpy default generator

        Returns:
            A ``[B K C]`` tensor of centroid features.
        """
        return torch_kmeans_cosine(
            f_backbone, num_objects, seed, self.max_iters, self.tol
        )


def torch_kmeans_euclidean(
    x, num_clusters: int, seed: int = None, max_iters=10, tol=1e-8
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
    idx = torch.from_numpy(rng.choice(N, size=(B, K), replace=True)).to(x.device)

    # [B K C] gather centroids from x using the sampled indexes
    centroids = x.detach().gather(dim=1, index=idx[:, :, None].expand(-1, -1, C))

    for i in range(max_iters):
        # [B N K] mean squared error between each point and each centroid
        mse = (centroids[:, None, :, :] - x[:, :, None, :]).pow_(2).sum(dim=-1)

        # [B N] assign the closest centroid to each sample
        idx = mse.argmin(dim=-1)

        # [B K] how many samples are assigned to each centroid?
        counts = (
            idx.new_zeros(B, K)
            .scatter_add_(dim=1, index=idx, src=idx.new_ones(B, N))
            .clamp_min(1)
        )

        # [B K C] for each centroid, sum all samples assigned to it, divide by count
        new_centr = torch.zeros_like(centroids)
        new_centr.scatter_add_(dim=1, index=idx[:, :, None].expand(-1, -1, C), src=x)
        new_centr.div_(counts[:, :, None])

        # Compute tolerance and exit early
        error = centroids.sub_(new_centr).abs_().max()
        centroids = new_centr
        if error.item() < tol:
            break

    return centroids


def torch_kmeans_cosine(x, num_clusters: int, seed: int = None, max_iters=10, tol=1e-8):
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
