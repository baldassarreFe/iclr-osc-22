"""
Self, cross, co, and slot attentions.

Attention modules implement the query-key-value projections,
the attention itself, and the output projections.

Block modules wrap an attention module with layer norm,
feed-forward layers and residual connections.
"""
from typing import Tuple

import numpy as np
import opt_einsum
import torch
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, Mlp
from torch import nn


class SelfAttention(nn.Module):
    """Self attention.

    Check the :func:`forward` method.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """Init.

        Args:
            dim:
            num_heads:
            qkv_bias:
            attn_drop:
            proj_drop:
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = np.sqrt(head_dim)

        self.proj_qkv = nn.Sequential(
            nn.Linear(dim, 3 * dim, bias=qkv_bias),
            Rearrange(
                "B L (three H C) -> three B L H C", three=3, H=num_heads, C=head_dim
            ),
        )

        # Made-up batch size and sequence lengths to precompute einops
        B, L = 64, 128
        BLHC = (B, L, num_heads, head_dim)
        BHLL = (B, num_heads, L, L)
        self.dot_fn = opt_einsum.contract_expression("bqhc, bkhc -> bhqk", BLHC, BLHC)
        self.out_fn = opt_einsum.contract_expression("bhqk, bkhc -> bqhc", BHLL, BLHC)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_out = nn.Sequential(
            Rearrange("B L H C -> B L (H C)"),
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: tensor of shape ``[B L C]``

        Returns:
            Tensor of shape ``[B L C]``.
        """
        # Keys K=L, queries K=L

        # [B K HC] -> [3 B K H C] -> ([B K H C], [B K H C], [B K H C])
        q, k, v = self.proj_qkv(x).unbind(dim=0)

        # ([B Q H C], [B K H C]) -> [B H Q K]
        dots = self.dot_fn(q, k)
        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # ([B H Q K], [B K H C]) -> [B Q H C]
        out = self.out_fn(attn, v)

        # [B Q H C] -> [B Q HC]
        out = self.proj_out(out)
        return out


class CrossAttention(nn.Module):
    """Cross attention.

    Diagram::

        TODO

    Check the :func:`forward` method.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """Init.

        Args:
            dim:
            num_heads:
            qkv_bias:
            attn_drop:
            proj_drop:
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = np.sqrt(head_dim)

        self.proj_q = nn.Sequential(
            nn.Linear(dim, dim, bias=qkv_bias),
            Rearrange("B Q (H C) -> B Q H C", H=num_heads, C=head_dim),
        )
        self.proj_kv = nn.Sequential(
            nn.Linear(dim, 2 * dim, bias=qkv_bias),
            Rearrange("B K (two H C) -> two B K H C", two=2, H=num_heads, C=head_dim),
        )

        # Made-up batch size and sequence lengths to precompute einops
        B, Q, K = 64, 128, 128
        BQHC = (B, Q, num_heads, head_dim)
        BKHC = (B, K, num_heads, head_dim)
        BHQK = (B, num_heads, Q, K)
        self.dot_fn = opt_einsum.contract_expression("bqhc, bkhc -> bhqk", BQHC, BKHC)
        self.out_fn = opt_einsum.contract_expression("bhqk, bkhc -> bqhc", BHQK, BKHC)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_out = nn.Sequential(
            Rearrange("B Q H C -> B Q (H C)"),
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop),
        )

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: query tensor of shape ``[B Q C]``
            ctx: key-value tensor of shape ``[B K C]``

        Returns:
            Tensor of shape ``[B Q C]``.
        """
        # x:   [B Q C] queries
        # ctx: [B K C] context (key-value)

        # [B Q HC] -> [B Q H C]
        q = self.proj_q(x).div(self.scale)

        # [B K HC] -> [2 B K H C] -> ([B K H C], [B K H C])
        k, v = self.proj_kv(ctx).unbind(dim=0)

        # ([B Q H C], [B K H C]) -> [B H Q K]
        dots = self.dot_fn(q, k)
        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # ([B H Q K], [B K H C]) -> [B Q H C]
        out = self.out_fn(attn, v)

        # [B Q H C] -> [B Q HC]
        out = self.proj_out(out)
        return out


class SlotAttention(nn.Module):
    """Slot attention.

    Diagram::

        TODO

    Check the :func:`forward` method.
    """

    def __init__(
        self,
        dim: int,
        pos_embed=None,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = None,
    ):
        """Init.

        Args:
            dim:
            pos_embed:
            iters:
            eps:
            hidden_dim:
        """
        super().__init__()
        self.num_iters = iters
        self.eps = eps
        self.scale = np.sqrt(dim)
        if hidden_dim is None:
            hidden_dim = dim

        if pos_embed is None or pos_embed == "none":
            pos_embed = nn.Identity()
        self.pos_embed = pos_embed

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # Dummy module to get the attn matrix with a pre-forward hook
        self.dot_prod_softmax = nn.Identity()

        self.gru = nn.GRUCell(dim, dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, slots: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            slots: query tensor of shape ``[B S C]``
            inputs: key-value tensor of shape ``[B N C]``

        Returns:
            Tensor of shape ``[B S C]``.
        """

        B, S, C = slots.shape
        B, N, C = inputs.shape

        inputs = self.pos_embed(inputs)
        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        for i in range(self.num_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.to_q(slots).div_(self.scale)
            dots = torch.einsum("bqd,bkd->bqk", q, k)
            attn = dots.softmax(dim=-2)
            attn, _ = self.dot_prod_softmax((attn, i))
            attn = attn + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum("bjd,bij->bid", v, attn)
            slots = self.gru(
                updates.reshape(-1, C),
                slots_prev.reshape(-1, C),
            ).reshape(B, -1, C)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


class SelfAttentionBlock(nn.Module):
    """Self attention block.

    Diagram::

         ┌───┤
         │   ▼
         │ norm
         │   │
         │   ▼
         │ attn
         │   │
         │   ▼
         └──►+
         ┌───┤
         │   ▼
         │ norm
         │   │
         │   ▼
         │  MLP
         │   │
         │   ▼
         └──►+
             ▼

    Check the :func:`forward` method.
    """

    # Reference implementation: timm.models.vision_transformer.Block
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        """Init.

        Args:
            dim:
            num_heads:
            mlp_ratio:
            qkv_bias:
            drop:
            attn_drop:
            drop_path:
            act_layer:
            norm_layer:
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(np.round(dim * mlp_ratio)),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: query tensor of shape ``[B L C]``

        Returns:
            Tensor of shape ``[B L C]``.
        """

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttentionBlock(nn.Module):
    """Cross attention block.

    Check the :func:`forward` method.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        """Init.

        Args:
            dim:
            num_heads:
            mlp_ratio:
            qkv_bias:
            drop:
            attn_drop:
            drop_path:
            act_layer:
            norm_layer:
        """
        super().__init__()
        self.norm1_x = norm_layer(dim)
        self.norm1_ctx = norm_layer(dim)
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(np.round(dim * mlp_ratio)),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: query tensor of shape ``[B Q C]``
            ctx: key-value tensor of shape ``[B K C]``

        Returns:
            Tensor of shape ``[B Q C]``.
        """

        x = x + self.drop_path(self.attn(self.norm1_x(x), self.norm1_ctx(ctx)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CoAttentionBlock(nn.Module):
    """Co-attention block, both a->b and b->a.

    Check the :func:`forward` method.
    """

    def __init__(self, *args, **kwargs):
        """Init.

        Args:
            *args: Same as :class:`CrossAttentionBlock`
            **kwargs: Same as :class:`CrossAttentionBlock`
        """
        super().__init__()
        self.cross_a_b = CrossAttentionBlock(*args, **kwargs)
        self.cross_b_a = CrossAttentionBlock(*args, **kwargs)

    def forward(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward.

        Args:
            a: tensor of shape ``[B N C]``
            b: tensor of shape ``[B M C]``

        Returns:
            Tensors with shapes:
            ``[B N C]`` (``a`` attending to ``b``),
            ``[B M C]`` (``b`` attending to ``a``).
        """

        new_a = self.cross_a_b(a, b)
        new_b = self.cross_b_a(b, a)
        return new_a, new_b
