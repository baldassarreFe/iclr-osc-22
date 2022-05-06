"""
Self attention.

Attention modules implement the query-key-value projections,
the attention itself, and the output projections.

Block modules wrap an attention module with layer norm,
feed-forward layers and residual connections.
"""

from typing import Callable, Type

import numpy as np
import opt_einsum
from einops.layers.torch import Rearrange
from timm.models.layers.drop import DropPath
from torch import Tensor, nn

from .utils import MLP, RegularSoftmax


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
        self.softmax = RegularSoftmax(p=attn_drop)
        self.out_fn = opt_einsum.contract_expression("bhqk, bkhc -> bqhc", BHLL, BLHC)

        self.proj_out = nn.Sequential(
            Rearrange("B L H C -> B L (H C)"),
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop),
        )

        self.init_weights()

    def init_weights(self):
        w = self.proj_qkv[0].weight
        val = np.sqrt(6.0 / float(w.shape[0] // 3 + w.shape[1]))
        nn.init.uniform_(w, -val, val)
        if self.proj_qkv[0].bias is not None:
            nn.init.zeros_(self.proj_qkv[0].bias)

        nn.init.xavier_uniform_(self.proj_out[1].weight)
        if self.proj_out[1].bias is not None:
            nn.init.zeros_(self.proj_out[1].bias)

    def forward(self, x: Tensor) -> Tensor:
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
        attn = self.softmax(dots)

        # ([B H Q K], [B K H C]) -> [B Q H C]
        out = self.out_fn(attn, v)

        # [B Q H C] -> [B Q HC]
        out = self.proj_out(out)
        return out


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
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
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
        self.mlp = MLP(
            in_features=dim,
            hidden_mult=mlp_ratio,
            out_features=dim,
            activation=act_layer,
            dropout=drop,
        )

        self.init_weights()

    def init_weights(self):
        # Norm layers already initialize weight=1 and bias=0
        # Other layers can take care of themselves.
        pass

    def forward(self, x: Tensor) -> Tensor:
        """Forward.

        Args:
            x: query tensor of shape ``[B L C]``

        Returns:
            Tensor of shape ``[B L C]``.
        """

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
