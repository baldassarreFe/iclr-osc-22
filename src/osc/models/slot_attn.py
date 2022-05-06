"""
Slot attention with optional fixed point trick.
"""
from functools import partial
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from osc.models.rollout import AttnInfo
from osc.models.utils import MLP, CompetitiveSoftmax


class SlotAttentionDecoder(nn.Module):
    """Slot attention.

    Diagram::

        TODO

    Check the :func:`forward` method.
    """

    def __init__(
        self,
        dim: int,
        pos_embed: Optional[Tensor] = None,
        pos_embed_drop: float = 0.0,
        num_layers: int = 3,
        mlp_ratio: float = 2.0,
        mlp_drop: float = 0.0,
        fixed_point: bool = False,
        bias_first: float = None,
    ):
        """Init.

        Args:
            dim:
            pos_embed: a tensor of shape ``[H W C]``
            pos_embed_drop:
            num_layers:
            mlp_ratio:
            mlp_drop:
            fixed_point:
            bias_first:
        """
        super().__init__()
        self.scale = np.sqrt(dim)
        self.num_layers = num_layers
        self.fixed_point = fixed_point
        if num_layers <= 1:
            raise ValueError(f"num_layers must be positive, not {num_layers}")

        # Context ops
        if pos_embed is not None:
            if pos_embed.shape[-1] != dim:
                raise ValueError(f"Inconsistent dims: {dim}, {pos_embed.shape}")
            pos_embed = nn.Parameter(pos_embed)
        self.pos_embed = pos_embed
        self.pos_drop = nn.Dropout(pos_embed_drop)
        self.proj_kv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2 * dim, bias=False),
            Rearrange("B K (two C) -> two B K C", two=2, C=dim),
        )

        # Slots ops
        self.proj_q = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim, bias=False))

        self.bias_first = bias_first
        self.softmax = CompetitiveSoftmax()
        self.gru = nn.GRUCell(dim, dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_mult=mlp_ratio,
            out_features=dim,
            dropout=mlp_drop,
            activation=nn.ReLU,
        )

        self.init_weights()

    def init_weights(self):
        # Note: pos_embed is not initialized, in case it's from a pretrained backbone
        # trunc_normal_(self.pos_embed, std=.02)

        w = self.proj_kv[1].weight
        val = np.sqrt(6.0 / float(w.shape[0] // 2 + w.shape[1]))
        nn.init.uniform_(w, -val, val)
        if self.proj_kv[1].bias is not None:
            nn.init.zeros_(self.proj_kv[0].bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    def forward(self, slots: Tensor, ctx: Tensor) -> Tensor:
        """Forward.

        Args:
            slots: query tensor of shape ``[B S C]``
            ctx: key-value tensor of shape ``[B H W C]``

        Returns:
            Tensor of shape ``[B S C]``.
        """
        if self.pos_embed is not None:
            ctx = self.add_pos_embed(ctx)

        # [B H W C] -> ([B HW C], [B HW C])
        k, v = self.proj_kv(ctx.flatten(-3, -2))
        k = k.div(self.scale)

        use_grad = torch.is_grad_enabled() and not self.fixed_point
        with torch.set_grad_enabled(use_grad):
            for i in range(self.num_layers - 1):
                slots = self.forward_slots(slots, k, v)
        slots = self.forward_slots(slots, k, v)
        return slots

    def add_pos_embed(self, ctx: Tensor):
        pe = self.pos_embed
        HW = ctx.shape[-3:-1]
        if HW != pe.shape[-3:-1]:
            # [H', W', D] -> [D, H', W'] -> [D, H, W] -> [H, W, D]
            pe = pe.permute(2, 0, 1).unsqueeze(0)
            pe = F.interpolate(pe, size=HW, mode="bicubic", align_corners=False)
            pe = pe.squeeze(0).permute(1, 2, 0)
        return self.pos_drop(ctx + pe)

    def forward_slots(self, slots: Tensor, k: Tensor, v: Tensor) -> Tensor:
        B, S, C = slots.shape
        q = self.proj_q(slots)
        dots = torch.einsum("bqd, bkd -> bqk", q, k)
        if self.bias_first is not None:
            dots[:, 0, :].add_(self.bias_first)
        attn = self.softmax(dots)
        updates = torch.einsum("bqk, bkd -> bqd", attn, v)
        slots = self.gru(updates.reshape(B * S, C), slots.reshape(B * S, C))
        slots = slots.reshape(B, S, C)
        slots = slots + self.mlp(self.norm_mlp(slots))
        return slots

    def register_attention_hooks(self, attns: List[AttnInfo]) -> List[RemovableHandle]:
        def softmax_hook(module, inputs, output, *, name: str, fixed_point):
            attns.append(AttnInfo(module=module, name=name, attn=output.detach()))

        # Use a hook on the softmax module
        handle = self.softmax.register_forward_hook(
            partial(softmax_hook, name="competitive", fixed_point=self.fixed_point)
        )
        return [handle]

    @staticmethod
    def rollout(attns: List[AttnInfo]) -> Tensor:
        # [B, S, K]
        return attns[-1].attn

    @staticmethod
    def attn_to_img(attn_info: AttnInfo, resolution: (int, int)) -> Tensor:
        return rearrange(
            attn_info.attn, "B S (K_h K_w) -> B S K_h K_w", K_h=resolution[0]
        )
