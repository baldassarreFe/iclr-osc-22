"""
Self attention.

Attention modules implement the query-key-value projections,
the attention itself, and the output projections.

Block modules wrap an attention module with layer norm,
feed-forward layers and residual connections.
"""

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import opt_einsum
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers.drop import DropPath
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from .rollout import AttnInfo
from .self_attn import SelfAttention
from .utils import MLP, RegularSoftmax


class CrossAttention(nn.Module):
    """Cross attention.

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
        self.softmax = RegularSoftmax(attn_drop)
        self.out_fn = opt_einsum.contract_expression("bhqk, bkhc -> bqhc", BHQK, BKHC)

        self.proj_out = nn.Sequential(
            Rearrange("B Q H C -> B Q (H C)"),
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop),
        )

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.proj_q[0].weight)
        if self.proj_q[0].bias is not None:
            nn.init.zeros_(self.proj_q[0].bias)

        w = self.proj_kv[0].weight
        val = np.sqrt(6.0 / float(w.shape[0] // 2 + w.shape[1]))
        nn.init.uniform_(w, -val, val)
        if self.proj_kv[0].bias is not None:
            nn.init.zeros_(self.proj_kv[0].bias)

        nn.init.xavier_uniform_(self.proj_out[1].weight)
        if self.proj_out[1].bias is not None:
            nn.init.zeros_(self.proj_out[1].bias)

    def forward(self, x: Tensor, ctx: Tensor) -> Tensor:
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
        attn = self.softmax(dots)

        # ([B H Q K], [B K H C]) -> [B Q H C]
        out = self.out_fn(attn, v)

        # [B Q H C] -> [B Q HC]
        out = self.proj_out(out)
        return out


class CrossAttentionBlock(nn.Module):
    """Cross attention block.

    Diagram::

             x   ctx
         ┌───┤    │
         │   ▼    │
         │ norm   │
         │   │    │
         │   ▼    │
         │ self   │
         │ attn   │
         │   │    │
         │   ▼    │
         └──►+    │
         ┌───┤    │
         │   ▼    │
         │ norm   │
         │   │    │
         │   ▼    │
         │ cross◄─┘
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

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        mlp_ratio: float = 4.0,
        mlp_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        skip_self_attn: bool = False,
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
            act_layer:
            norm_layer:
        """
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.skip_self_attn = skip_self_attn
        if not skip_self_attn:
            self.self_norm = norm_layer(dim)
            self.self_attn = SelfAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )

        self.cross_norm = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.mlp_norm = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_mult=mlp_ratio,
            out_features=dim,
            activation=act_layer,
            dropout=mlp_drop,
        )

        self.init_weights()

    def init_weights(self):
        # Norm layers already initialize weight=1 and bias=0
        # Other layers can take care of themselves.
        pass

    def forward(self, x: Tensor, ctx: Tensor) -> Tensor:
        """Forward.

        Args:
            x: query tensor of shape ``[B Q C]``
            ctx: key-value tensor of shape ``[B K C]``

        Returns:
            Tensor of shape ``[B Q C]``.
        """
        if not self.skip_self_attn:
            x = x + self.drop_path(self.self_attn(self.self_norm(x)))
        x = x + self.drop_path(self.cross_attn(self.cross_norm(x), ctx))
        x = x + self.drop_path(self.mlp(self.mlp_norm(x)))
        return x


class CrossAttentionDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        pos_embed: Optional[Tensor] = None,
        pos_embed_drop: float = 0.0,
        num_layers: int = 3,
        reuse_layers: bool = False,
        num_heads: int = 4,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        mlp_ratio: float = 4.0,
        mlp_drop: float = 0.0,
        qkv_bias: bool = False,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ):
        """Init.

        Args:
            dim:
            pos_embed: a tensor of shape ``[H W C]``
            num_heads:
            num_layers:
            reuse_layers:
            proj_drop:
            attn_drop:
            drop_path:
            mlp_ratio:
            act_layer:
            norm_layer:
        """
        super().__init__()

        if pos_embed is not None:
            if pos_embed.shape[-1] != dim:
                raise ValueError(f"Inconsistent dims: {dim}, {pos_embed.shape}")
            pos_embed = nn.Parameter(pos_embed)
        self.pos_embed = pos_embed
        self.pos_drop = nn.Dropout(pos_embed_drop)
        self.ctx_norm = norm_layer(dim)

        # B N C -> B Q C
        blocks = [
            CrossAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                mlp_drop=mlp_drop,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                skip_self_attn=False,
            )
            for _ in range(num_layers)
        ]
        if reuse_layers:
            blocks = num_layers * [blocks[0]]
        self.blocks = nn.ModuleList(blocks)

        self.init_weights()

    def init_weights(self):
        # Note: pos_embed is not initialized, in case it's from a pretrained backbone
        # trunc_normal_(self.pos_embed, std=.02)
        pass

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    def forward(self, slots: Tensor, ctx: Tensor) -> Tensor:
        """Forward.

        Args:
            slots: query tensor of shape ``[B S C]``
            ctx: key-value tensor of shape ``[B H W C]``

        Returns:
            Slot tensor of shape ``[B S C]``.
        """
        if self.pos_embed is not None:
            ctx = self.add_pos_embed(ctx)

        # [B H W C] -> [B HW C]
        ctx = self.ctx_norm(ctx.flatten(-3, -2))

        for block in self.blocks:
            slots = block(slots, ctx)
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

    def register_attention_hooks(self, attns: List[AttnInfo]) -> List[RemovableHandle]:
        return register_attention_hooks(self, attns)

    @staticmethod
    def rollout(
        attns: List[AttnInfo],
        head_reduction: Union[str, Callable[[Tensor], Tensor]] = "mean",
    ) -> Tensor:
        """Rollout.

        Args:
            attns: list of self and cross attentions, shapes
                ``[B, head, S, S]`` and ``[B, head, S, K]``
            head_reduction:
                or to the input, shape ``[B, K, K']``

        Returns:
            Rollout of slots to the context, shape ``[B, S, K']``
        """
        if len(attns) < 2 or len(attns) % 2 != 0:
            raise ValueError("Invalid length of self-cross attentions")
        if any(a.extra["stage"] != "self" for a in attns[::2]):
            raise ValueError("Invalid sequence of self attentions")
        if any(a.extra["stage"] != "cross" for a in attns[1::2]):
            raise ValueError("Invalid sequence of cross attentions")

        attns: List[Tensor] = [a.attn for a in attns]
        B, _, S, K = attns[1].shape
        device = attns[1].device

        # Reduce heads: mean or max
        if head_reduction in {"mean", "max"}:
            head_reduction = Reduce("B head Q K -> B Q K", head_reduction)
        attns = [head_reduction(a) for a in attns]

        B, S, K = attns[1].shape
        eye = torch.eye(S, S, device=device)
        roll_slot_slot = torch.eye(S, S, device=device).expand(B, S, S)
        roll_slot_ctx = torch.zeros(B, S, K, device=device)

        for stage, attn in enumerate(attns):
            if stage % 2 == 0:
                # Self attention
                attn = attn + eye
                roll_slot_slot = torch.bmm(attn, roll_slot_slot)
                roll_slot_ctx = torch.bmm(attn, roll_slot_ctx)
            else:
                # Cross attention
                # roll_slot_slot = roll_slot_slot
                roll_slot_ctx = attn + roll_slot_ctx

        return roll_slot_ctx

    @staticmethod
    def attn_to_img(
        attn_info: AttnInfo,
        resolution: (int, int),
        head_reduction: Union[str, Callable[[Tensor], Tensor]] = "mean",
    ) -> Optional[Tensor]:
        # Note: for cross attention only
        if attn_info.extra["stage"] != "cross":
            return None
        if head_reduction in {"mean", "max"}:
            head_reduction = Reduce("B h S K -> B S K", head_reduction)
        attn = head_reduction(attn_info.attn)
        return rearrange(attn, "B S (K_h K_w) -> B S K_h K_w", K_h=resolution[0])


class CrossAttentionPooling(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int = 3,
        reuse_layers: bool = False,
        num_heads: int = 4,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        mlp_ratio: float = 4.0,
        mlp_drop: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        qkv_bias: bool = False,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ):
        """Init.

        Args:
            dim:
            num_heads:
            num_layers:
            reuse_layers:
            proj_drop:
            attn_drop:
            drop_path:
            mlp_ratio:
            act_layer:
            norm_layer:
        """
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(dim))
        self.ctx_norm = norm_layer(dim)

        # B N C -> B S C
        blocks = [
            CrossAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                mlp_drop=mlp_drop,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                skip_self_attn=True,
            )
            for _ in range(num_layers)
        ]
        if reuse_layers:
            blocks = num_layers * [blocks[0]]
        self.blocks = nn.ModuleList(blocks)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.cls_token, std=1e-6)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token"}

    def forward(self, ctx: Tensor) -> Tensor:
        """Forward.

        Args:
            ctx: key-value tensor of shape ``[B N C]``

        Returns:
            CLS token of shape ``[B C]``.
        """
        ctx = self.ctx_norm(ctx)
        cls_token = self.cls_token.expand(ctx.shape[0], 1, -1)
        for block in self.blocks:
            cls_token = block(cls_token, ctx)
        return cls_token.squeeze(1)

    def register_attention_hooks(self, attns: List[AttnInfo]) -> List[RemovableHandle]:
        return register_attention_hooks(self, attns)

    @staticmethod
    def rollout(
        attns: List[AttnInfo],
        head_reduction: Union[str, Callable[[Tensor], Tensor]] = "mean",
    ) -> Tensor:
        """Rollout.

        Args:
            attns: list of pooling cross attentions, shape ``[B, head, 1, K]``
            head_reduction:

        Returns:
            Rollout of ``cls_token`` to the context or input, shape ``[B, K']``
        """
        attns: List[Tensor] = [a.attn for a in attns]
        B, _, Q, K = attns[0].shape
        device = attns[0].device
        if Q != 1:
            raise ValueError(f"Cross attention pooling requires Q=1, not {Q}")

        # Reduce heads: mean or max
        if head_reduction in {"mean", "max"}:
            head_reduction = Reduce("B head Q K -> B Q K", head_reduction)
        attns = [head_reduction(a) for a in attns]

        rollout_pool_ctx = torch.zeros(B, 1, K, device=device)
        for attn in attns:
            rollout_pool_ctx = attn + rollout_pool_ctx

        return rollout_pool_ctx

    @staticmethod
    def attn_to_img(
        attn_info: AttnInfo,
        resolution: (int, int),
        head_reduction: Union[None, str, Callable[[Tensor], Tensor]] = "mean",
    ) -> Tensor:
        attn = attn_info.attn
        if head_reduction is not None:
            if head_reduction in {"mean", "max"}:
                head_reduction = Reduce("B h Q K -> B Q K", head_reduction)
            attn = head_reduction(attn)
            attn = rearrange(attn, "B () (K_h K_w) -> B K_h K_w", K_h=resolution[0])
        else:
            attn = rearrange(
                attn, "B head () (K_h K_w) -> B head K_h K_w", K_h=resolution[0]
            )
        return attn


def register_attention_hooks(
    decoder: Union[CrossAttentionDecoder, CrossAttentionPooling], attns: List[AttnInfo]
) -> List[RemovableHandle]:
    def softmax_hook(module, inputs, output, *, name: str, extra: Dict[str, Any]):
        attns.append(
            AttnInfo(module=module, name=name, attn=output.detach(), extra=extra)
        )

    handles = []
    for name, module in decoder.named_modules():
        if isinstance(module, SelfAttention):
            # Use a hook on the softmax module
            hook = partial(softmax_hook, name=name, extra={"stage": "self"})
            handle = module.softmax.register_forward_hook(hook)
            handles.append(handle)
        elif isinstance(module, CrossAttention):
            # Use a hook on the softmax module
            hook = partial(softmax_hook, name=name, extra={"stage": "cross"})
            handle = module.softmax.register_forward_hook(hook)
            handles.append(handle)
    return handles
