"""
Top-k competitive attention.

Attention modules implement the query-key-value projections,
the attention itself, and the output projections.

Block modules wrap an attention module with layer norm,
feed-forward layers and residual connections.
"""
from typing import Callable, Optional, Tuple, Type, Union

import numpy as np
import opt_einsum
import timm.models.vision_transformer
import torch
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, Mlp
from torch import Tensor, nn

from .utils import CompetitiveSoftmax


class TopkCompetitiveAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: Union[bool, Tuple[bool, bool, bool]] = False,
        max_queries: int = -1,
        keep_first: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = np.sqrt(head_dim)
        if not isinstance(qkv_bias, tuple):
            qkv_bias = 3 * (qkv_bias,)

        self.max_queries = max_queries
        self.keep_first = keep_first

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias[0])
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias[1])
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias[2])
        self.split_heads = Rearrange("B L (H C) -> B L H C", H=num_heads, C=head_dim)

        # Made-up batch size and sequence lengths to precompute einops
        B = 4
        Q = max_queries if max_queries > 0 else 20
        K = 49
        BQHC = (B, Q, num_heads, head_dim)
        BKHC = (B, K, num_heads, head_dim)
        BHQK = (B, num_heads, Q, K)
        self.dot_fn = opt_einsum.contract_expression("bqhc, bkhc -> bhqk", BQHC, BKHC)

        # Competitive attention, i.e. softmax over query axis
        self.softmax = CompetitiveSoftmax(attn_drop)
        self.out_fn = opt_einsum.contract_expression("bhqk, bkhc -> bqhc", BHQK, BKHC)

        self.proj_out = nn.Sequential(
            Rearrange("B Q H C -> B Q (H C)"),
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop),
        )

    def init_weights(self):
        for proj in [self.proj_q, self.proj_k, self.proj_v, self.proj_out[1]]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward.

        Args:
            q: queries of shape ``[B Q C]``
            k: keys of shape ``[B K C]``
            v: values of shape ``[B K C]``
            max_queries:

        Returns:
            Attention output ``[B Q C]``, where ``Q`` is either the original ``Q`` or
            equal to ``max_queries``. If ``max_queries>0``, also return the indices of
            the queries that were kept ``[B max_queries]``.
        """
        B, Q, _ = q.shape
        B, K, _ = k.shape

        # q: [B Q HC] -> [B Q H C]
        q = self.split_heads(self.proj_q(q))
        # k: [B K HC] -> [B K H C]
        k = self.split_heads(self.proj_k(k))
        # value: [B K HC] -> [B K H C]
        v = self.split_heads(self.proj_v(v))

        # ([B Q H C], [B K H C]) -> [B H Q K]
        dots = self.dot_fn(q.div(self.scale), k)
        dots, keep = select_queries(
            dots,
            max_queries=self.max_queries,
            keep_first=self.keep_first,
        )

        # [B H Q K]
        attn = self.softmax(dots)
        # ([B H Q K], [B K H C]) -> [B Q H C]
        out = self.out_fn(attn, v)
        # [B Q H C] -> [B Q HC]
        out = self.proj_out(out)

        return out, keep


class TopkCompetitiveAttentionBlock(nn.Module):
    """Topk competitive attention block.

    Diagram::

        TODO

    Check the :func:`forward` method.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: Union[bool, Tuple[bool, bool, bool]] = False,
        max_queries: int = -1,
        keep_first: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        mlp_ratio=4.0,
        mlp_drop=0.0,
        mlp_act: Type[nn.Module] = nn.GELU,
        drop_path=0.0,
    ):
        """Init.

        Args:
            dim:
            num_heads:
            mlp_ratio:
            qkv_bias:
            proj_drop:
            attn_drop:
            drop_path:
            mlp_act:
            norm_layer:
        """
        super().__init__()
        self.topk_attn = TopkCompetitiveAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            max_queries=max_queries,
            keep_first=keep_first,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm1 = norm_layer(dim)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(np.round(dim * mlp_ratio)),
            act_layer=mlp_act,
            drop=mlp_drop,
        )
        self.norm2 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Forward.

        Args:
            q: queries of shape ``[B Q C]``
            k: keys of shape ``[B K C]``
            v: values of shape ``[B K C]``

        Returns:
            Tensor of shape ``[B Q C]``, where ``Q`` is either the original ``Q`` or
            equal to ``max_queries``. If ``max_queries>0``, also return the indices of
            the queries that were kept ``[B max_queries]``.
        """
        q_0 = q
        q, keep = self.topk_attn(q, k, v)
        if keep is not None:
            # [B Q C] -> [B max_queries C]
            keep = keep[:, :, None].expand(-1, -1, q.shape[-1])
            q_0 = torch.gather(q_0, dim=-2, index=keep)

        q = q_0 + self.drop_path(self.norm1(q))
        q = q + self.drop_path(self.norm2(self.mlp(q)))
        return q


class TopkCompetitiveInstanceAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: Union[bool, Tuple[bool, bool, bool]] = False,
        max_queries: int = -1,
        keep_first: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = np.sqrt(head_dim)
        if not isinstance(qkv_bias, tuple):
            qkv_bias = 3 * (qkv_bias,)

        self.max_queries = max_queries
        self.keep_first = keep_first

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias[0])
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias[1])
        self.proj_feats = nn.Linear(dim, dim, bias=qkv_bias[2])
        self.proj_enc = nn.Linear(dim, dim, bias=qkv_bias[2])
        self.split_heads = Rearrange("B L (H C) -> B L H C", H=num_heads, C=head_dim)

        # Made-up batch size and sequence lengths to precompute einops
        B = 4
        Q = max_queries if max_queries > 0 else 20
        K = 14 * 14
        BQHC = (B, Q, num_heads, head_dim)
        BKHC = (B, K, num_heads, head_dim)
        BHQK = (B, num_heads, Q, K)
        self.dot_fn = opt_einsum.contract_expression("bqhc, bkhc -> bhqk", BQHC, BKHC)

        # Competitive attention, i.e. softmax over query axis
        self.softmax = CompetitiveSoftmax(attn_drop)
        self.out_fn = opt_einsum.contract_expression("bhqk, bkhc -> bqhc", BHQK, BKHC)

        self.proj_out_feats = nn.Sequential(
            Rearrange("B Q H C -> B Q (H C)"),
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop),
        )
        self.proj_out_enc = nn.Sequential(
            Rearrange("B Q H C -> B Q (H C)"),
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop),
        )

    def forward(
        self,
        slots: Tensor,
        slots_enc: Tensor,
        ctx: Tensor,
        ctx_enc: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Forward."""
        B, Q, _ = slots.shape
        B, K, _ = ctx.shape

        # q: [B Q HC] -> [B Q H C]
        q = self.split_heads(self.proj_q(slots + slots_enc))
        # k: [B K HC] -> [B K H C]
        k = self.split_heads(self.proj_k(ctx + ctx_enc))
        # value: [B K HC] -> [B K H C]
        ctx = self.split_heads(self.proj_feats(ctx))
        ctx_enc = self.split_heads(self.proj_enc(ctx_enc))

        # ([B Q H C], [B K H C]) -> [B H Q K]
        dots = self.dot_fn(q.div(self.scale), k)
        dots, keep = select_queries(dots, self.max_queries, self.keep_first)

        # [B H Q K]
        attn = self.softmax(dots)
        # ([B H Q K], [B K H C]) -> [B Q H C] -> [B Q HC]
        slots = self.proj_out_feats(self.out_fn(attn, ctx))
        slots_enc = self.proj_out_enc(self.out_fn(attn, ctx_enc))

        return slots, slots_enc, keep


class TopkCompetitiveInstanceAttentionBlock(nn.Module):
    """Topk competitive instance attention block.

    Diagram::

        TODO

    Check the :func:`forward` method.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: Union[bool, Tuple[bool, bool, bool]] = False,
        max_queries: int = -1,
        keep_first: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        mlp_ratio=4.0,
        mlp_drop=0.0,
        mlp_act: Type[nn.Module] = nn.GELU,
        drop_path=0.0,
    ):
        """Init.

        Args:
            dim:
            num_heads:
            mlp_ratio:
            qkv_bias:
            proj_drop:
            attn_drop:
            drop_path:
            mlp_act:
            norm_layer:
        """
        super().__init__()
        self.topk_attn = TopkCompetitiveInstanceAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            max_queries=max_queries,
            keep_first=keep_first,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.norm_slots1 = norm_layer(dim)
        self.mlp_slots = Mlp(
            in_features=dim,
            hidden_features=int(np.round(dim * mlp_ratio)),
            act_layer=mlp_act,
            drop=mlp_drop,
        )
        self.norm_slots2 = norm_layer(dim)

        self.norm_enc1 = norm_layer(dim)
        self.mlp_enc = Mlp(
            in_features=dim,
            hidden_features=int(np.round(dim * mlp_ratio)),
            act_layer=mlp_act,
            drop=mlp_drop,
        )
        self.norm_enc2 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.init_weights()

    def init_weights(self):
        self.apply(timm.models.vision_transformer._init_vit_weights)

    def forward(
        self,
        slots: Tensor,
        slots_enc: Tensor,
        ctx_feats: Tensor,
        ctx_enc: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Forward."""
        B, Q, C = slots.shape

        attn_slots, attn_slots_enc, keep = self.topk_attn(
            self.norm_slots1(slots),
            self.norm_enc1(slots_enc),
            self.norm_slots1(ctx_feats),
            self.norm_enc1(ctx_enc),
        )
        if keep is not None:
            # [B Q C] -> [B max_queries C]
            keep = keep[:, :, None].expand(-1, -1, C)
            slots = torch.gather(slots, dim=-2, index=keep)
            slots_enc = torch.gather(slots_enc, dim=-2, index=keep)

        slots = slots + self.drop_path(attn_slots)
        slots = slots + self.drop_path(self.mlp_slots(self.norm_slots2(slots)))

        slots_enc = slots_enc + self.drop_path(attn_slots_enc)
        slots_enc = slots_enc + self.drop_path(self.mlp_enc(self.norm_enc2(slots_enc)))

        return slots, slots_enc


class TopkCompetitiveInstanceAttentionDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        pos_embed=None,
        pos_embed_drop: float = 0.0,
        background_token: bool = True,
        num_objects: int = 11,
        num_heads: int = 4,
        num_layers: int = 3,
        reuse_layers: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        mlp_ratio: float = 4.0,
        mlp_drop: float = 0.0,
        mlp_act: Type[nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ):
        """Init.

        Args:
            dim:
            background_token:
            num_heads:
            num_layers:
            reuse_layers:
            proj_drop:
            attn_drop:
            drop_path:
            mlp_ratio:
            mlp_act:
            norm_layer:
        """
        super().__init__()

        self.register_parameter("pos_embed", nn.Parameter(pos_embed))
        self.pos_drop = nn.Dropout(pos_embed_drop)

        self.background_token = background_token

        # B N C -> B N C
        blocks = [
            TopkCompetitiveInstanceAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=False,
                keep_first=background_token,
                max_queries=num_objects,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                mlp_ratio=mlp_ratio,
                mlp_drop=mlp_drop,
                mlp_act=mlp_act,
                norm_layer=norm_layer,
                drop_path=drop_path,
            )
            for _ in range(num_layers)
        ]
        if reuse_layers:
            blocks = num_layers * [blocks[0]]
        self.blocks = nn.ModuleList(blocks)

        self.init_weights()

    def init_weights(self):
        # Note: this does not reset pos_embed, in case it's from a pretrained backbone
        # noinspection PyProtectedMember
        self.apply(timm.models.vision_transformer._init_vit_weights)

    def forward(self, slots: Tensor, ctx: Tensor) -> Tensor:
        """Forward.

        Args:
            slots: query tensor of shape ``[B E C]``
            ctx: key-value tensor of shape ``[B N C]``

        Returns:
            Tensor of shape ``[B E C]``, not layer-normalized
        """
        B, N, C = ctx.shape
        if self.pos_embed is not None:
            ctx_enc = self.pos_drop(self.pos_embed.expand(B, -1, -1))
        else:
            ctx_enc = torch.zeros_like(ctx)

        # TODO deterministic behavior if not self.training
        B, E, C = slots.shape
        if self.pos_embed is not None:
            slots_enc = torch.randint(0, N, (B, E), device=slots.device)
            slots_enc = self.pos_drop(self.pos_embed[slots_enc])
        else:
            slots_enc = torch.zeros_like(slots)

        for block in self.blocks:
            if self.background_token:
                slots_enc[:, 0, :].zero_()
            slots, slots_enc = block(slots, slots_enc, ctx, ctx_enc)

        # TODO Does it make sense to have a final x+y+MLP(LN(x + y)) before return?
        return slots + slots_enc


def select_queries(
    dots: Tensor, max_queries: int, keep_first: bool
) -> Tuple[Tensor, Optional[Tensor]]:
    B, H, Q, K = dots.shape
    if max_queries <= 0 or max_queries >= Q:
        return dots, None

    # keep: [B, max_queries]
    if not keep_first:
        keep = (
            dots.detach()
            .softmax(-2)  # axis: Q
            .amax(dim=(-3, -1))  # axes: H, K
            .topk(max_queries, dim=-1, sorted=False)  # axis: Q
            .indices
        )
    else:
        keep = (
            dots.detach()[:, :, 1:, :]
            .softmax(-2)  # axis: Q
            .amax(dim=(-3, -1))  # axes: H, K
            .topk(max_queries, dim=-1, sorted=False)  # axis: Q
            .indices
        )
        keep = torch.cat(
            [
                keep.new_zeros((B, 1)),
                keep.add_(1),
            ],
            dim=-1,  # axis: Q
        )

    # dots: [B H Q K] -> [B H max_queries K]
    dots = torch.gather(
        dots,
        dim=-2,
        index=keep[:, None, :, None].expand(-1, H, -1, K),
    )
    return dots, keep
