import torch
from torch import nn as nn


class SlotAttention(nn.Module):
    def __init__(
        self,
        num_slots,
        dim,
        pos_embed=None,
        iters=3,
        eps=1e-8,
        hidden_dim=128,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        hidden_dim = max(dim, hidden_dim)

        if pos_embed is None or pos_embed == "none":
            pos_embed = nn.Identity()
        self.pos_embed = pos_embed

        self.slots_mu = nn.Parameter(torch.zeros(dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)
        self.dot_prod_softmax = SlotAttention.DotProdSoftmax()

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.slots_mu)
        nn.init.normal_(self.slots_log_sigma)

    # noinspection PyPep8Naming
    def _get_slots(self, B, K: int = None, seed: int = None):
        device = self.slots_mu.device
        C = self.slots_mu.shape[-1]
        if K is None:
            K = self.num_slots

        if self.training:
            # Use global rng unless seed is explicitly given
            rng = None if seed is None else torch.Generator(device).manual_seed(seed)
            # Sample B*K independent vectors of length C
            slots = torch.randn(B, K, C, device=device, generator=rng)
            slots = self.slots_mu + self.slots_log_sigma.exp() * slots

        else:
            # Use a one-time rng with fixed seed unless seed is explicitly given
            rng = torch.Generator(device).manual_seed(
                seed if seed is not None else 18327415066407224732
            )
            # Sample K independent vectors of length C, then reuse them for all B images
            slots = torch.randn(K, C, device=device, generator=rng)
            slots = self.slots_mu + self.slots_log_sigma.exp() * slots
            slots = slots.expand(B, -1, -1)

        return slots

    def forward(self, inputs: torch.Tensor, num_slots: int = None, seed: int = None):
        B, N, C = inputs.shape
        slots = self._get_slots(B, num_slots, seed)

        inputs = self.pos_embed(inputs)
        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        for i in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
            attn = dots.softmax(dim=1)
            attn = self.dot_prod_softmax(attn, i)
            attn = attn + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum("bjd,bij->bid", v, attn)
            slots = self.gru(updates.reshape(-1, C), slots_prev.reshape(-1, C)).reshape(
                B, -1, C
            )
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots

    class DotProdSoftmax(nn.Module):
        """Dummy module, allows inserting a pre-forward hook to get the attn matrix"""

        # noinspection PyUnusedLocal,PyMethodMayBeStatic
        def forward(self, attn: torch.Tensor, iter_idx: int) -> torch.Tensor:
            return attn
