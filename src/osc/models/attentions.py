import numpy as np
import opt_einsum
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, Mlp
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
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

    def forward(self, x):
        # x: [B L C]
        # K = Q = L

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
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
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

    def forward(self, x, ctx):
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


class SelfAttentionBlock(nn.Module):
    # Reference implementation: timm.models.vision_transformer.Block
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
        mlp_hidden_dim = int(np.round(dim * mlp_ratio))
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttentionBlock(nn.Module):
    # Reference implementation: timm.models.vision_transformer.Block
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
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(np.round(dim * mlp_ratio))
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, ctx):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(ctx)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CoAttentionBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        """

        Args:
            *args: Same as :class:`CrossAttentionBlock`
            **kwargs: Same as :class:`CrossAttentionBlock`
        """
        super().__init__()
        self.cross_a_b = CrossAttentionBlock(*args, **kwargs)
        self.cross_b_a = CrossAttentionBlock(*args, **kwargs)

    def forward(self, a, b):
        new_a = self.cross_a_b(a, b)
        new_b = self.cross_b_a(b, a)
        return new_a, new_b
