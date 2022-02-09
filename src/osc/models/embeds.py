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
    """Dynamic number of sampled object tokens, shape ``[S C]``"""

    def __init__(self, embed_dim):
        super(SampledObjectTokens, self).__init__()
        self.mu = nn.Parameter(torch.zeros(embed_dim))
        self.log_sigma = nn.Parameter(torch.zeros(embed_dim))
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.slots_mu)
        nn.init.normal_(self.slots_log_sigma)

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
