import numpy as np
import timm.models
import torch
from torch import Tensor
from torch import nn as nn

from osc.models.kmeans import torch_kmeans_cosine, torch_kmeans_euclidean
from osc.utils import l2_normalize

EVAL_SEED = 18327415066407224732


class LearnedObjectTokens(nn.Module):
    """Fixed number of learnable object tokens, shape ``[S C]``"""

    def __init__(self, embed_dim: int, num_objects: int):
        super(LearnedObjectTokens, self).__init__()
        self.tokens = nn.Parameter(torch.zeros((num_objects, embed_dim)))
        self.init_weights()

    def init_weights(self):
        timm.models.layers.trunc_normal_(self.tokens, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"tokens"}

    def forward(self, batch_size: int):
        return self.tokens.expand(batch_size, -1, -1)


class NormalObjectTokens(nn.Module):
    """Dynamically sample tokens from a normal object tokens, shape ``[S C]``"""

    def __init__(self, embed_dim: int, num_objects: int, eval_seed: int = EVAL_SEED):
        super(NormalObjectTokens, self).__init__()
        self.eval_seed = eval_seed
        self.embed_dim = embed_dim
        self.num_objects = num_objects
        self.register_buffer("dummy", torch.zeros([]), persistent=False)

    def forward(self, batch_size: int, num_objects: int = None):
        if num_objects is None:
            num_objects = self.num_objects
        if self.training:
            return self.forward_train(batch_size, num_objects)
        else:
            return self.forward_eval(batch_size, num_objects)

    def forward_train(self, batch_size: int, num_objects: int):
        device = self.dummy.device
        shape = (batch_size, num_objects, self.embed_dim)
        return torch.randn(shape, generator=None, device=device)

    def forward_eval(self, batch_size: int, num_objects: int):
        device = self.dummy.device
        shape = (num_objects, self.embed_dim)
        generator = torch.Generator(device).manual_seed(self.eval_seed)
        tokens = torch.randn(shape, generator=generator, device=device)
        return tokens.expand(batch_size, -1, -1)


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
                ``K`` Gaussian chosen uniformly at random with replacement.
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

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"mu", "log_sigma"}

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

        # Sample from a normal gaussian
        if True or self.training:
            # Sample B*S independent vectors of length D
            slots = torch.randn(B, S, D, device=device, generator=rng)
        else:
            # Sample S independent vectors of length D, then reuse them for all B images
            slots = torch.randn(1, S, D, device=device, generator=rng).expand(B, -1, -1)

        # If the parameters describe a single component, use that single mu and sigma.
        # Otherwise, uniformly sample S (mu, sigma) pairs out of the K learned pairs.
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


class PatchObjectTokens(nn.Module):
    def __init__(self, num_objects: int, eval_seed: int = EVAL_SEED):
        super(PatchObjectTokens, self).__init__()
        self.eval_seed = eval_seed
        self.num_objects = num_objects

    def forward(self, x: Tensor, num_objects: int = None):
        if num_objects is None:
            num_objects = self.num_objects
        if self.training:
            return self.forward_train(x, num_objects)
        else:
            return self.forward_eval(x, num_objects)

    def forward_train(self, x: Tensor, num_objects: int):
        B, L, C = x.shape
        idx = torch.randint(0, L, (B, num_objects), generator=None, device=x.device)
        return torch.gather(x, dim=-2, index=idx[:, :, None].expand(-1, -1, C))

    def forward_eval(self, x: Tensor, num_objects: int):
        B, L, C = x.shape
        generator = torch.Generator(x.device).manual_seed(self.eval_seed)
        idx = torch.randint(0, L, (num_objects,), generator=generator, device=x.device)
        return torch.gather(x, dim=-2, index=idx[None, :, None].expand(B, -1, C))


class KmeansCosineObjectTokens(nn.Module):
    def __init__(self, num_objects: int):
        super(KmeansCosineObjectTokens, self).__init__()
        self.max_iters = 20
        self.tol = 1e-6
        self.num_objects = num_objects

    def forward(self, x: Tensor, num_objects: int = None, seed=None):
        """K-means clustering to initialize object tokens.

        Args:
            x: feature tensor, shape``[B H W C]``. It will be L2-normalized internally.
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
            x: feature tensor, shape``[B H W C]``
            num_objects: number of objects
            seed: int seed, leave empty to use numpy default generator

        Returns:
            A ``[B K C]`` tensor of centroid features.
        """
        if num_objects is None:
            num_objects = self.num_objects
        return torch_kmeans_euclidean(x, num_objects, seed, self.max_iters, self.tol)
