"""
Customized VisionTransformer from timm.
"""
from functools import partial
from typing import Callable, List, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import reduce
from einops.layers.torch import Reduce
from timm.models.helpers import checkpoint_seq
from timm.models.layers.patch_embed import PatchEmbed
from timm.models.vision_transformer import Block, VisionTransformer
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from .rollout import AttnInfo


class MyVisionTransformer(VisionTransformer):
    """Custom version of VisionTransformer from timm.

    Modifications:
    - Allow variable-sized input images by resizing the positional embedding
    - Always use CLS token (not really used but pretrained models have it
      and allowing the option to not have it would make embed resizing harder)
    - Layer norm of output is disabled by default
    - ``forward_features`` splits the CLS token and reshapes patch tokens as a 2D grid
    - Method for getting the positional embedding as a 2D grid
    - Method for registering hooks that record attentions
    - Method for computing attention rollout given the attention maps

    """

    def __init__(
        self,
        img_size: (int, int) = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        pos_embed_drop: float = 0.0,
        global_pool: str = "token",
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        output_norm: bool = False,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            num_classes=0,
            global_pool=global_pool,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            embed_layer=MyPatchEmbed,
        )
        # timm implementation uses `drop_rate` for both pos_drop and proj_drop
        self.pos_drop.p = pos_embed_drop
        # timm implementation always applies layer norm at the output
        self.output_norm = output_norm

    def forward(self, images):
        """Forward through the transformer blocks, skip pooling and classifier.

        Args:
            images: tensor of shape ``[B, 3, H*P, W*P]``, where ``P`` is the patch size.

        Returns:
            A tuple of global and patch features. Global features have shape ``[B D]``.
            Patch features have shape``[B H W D]``.
        """
        # [B, 3, H*P, W*P] -> [B, H, W, D]
        x = self.patch_embed(images)
        B, H, W, D = x.shape
        x = x.reshape(B, H * W, D)

        if self.global_pool == "token":
            x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = self.add_pos_embed(x, H, W)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)

        if self.output_norm:
            x = self.norm(x)

        # Produce global representation
        if self.global_pool == "token":
            cls = x[:, 0, :]
            x = x[:, 1:, :]
        else:
            cls = x.mean(-2)

        return cls, x.reshape(B, H, W, D)

    def add_pos_embed(self, x: Tensor, H: int, W: int):
        pe = self.pos_embed

        if x.shape[-2] != pe.shape[-2]:
            if self.global_pool == "token":
                cls = pe[:, :1, :]
                pe = pe[:, 1:, :]

            # [H'W', D] -> [D, H', W'] -> [D, H, W] -> [H, W, D] -> [HW, D]
            D = pe.shape[-1]
            pe = pe.permute(0, 2, 1).reshape(1, D, *self.patch_embed.grid_size)
            pe = F.interpolate(pe, size=(H, W), mode="bicubic", align_corners=False)
            pe = pe.reshape(1, D, H * W).permute(0, 2, 1)

            if self.global_pool == "token":
                # noinspection PyUnboundLocalVariable
                pe = torch.cat((cls, pe), dim=1)

        return self.pos_drop(x + pe)

    def forward_features(self, x):
        raise NotImplementedError("Not implemented in subclass")

    def forward_head(self, x, pre_logits: bool = False):
        raise NotImplementedError("Not implemented in subclass")

    def get_pos_embed(self):
        """Positional embedding without CLS token reshaped as ``[H, W, D]``"""
        pe = self.pos_embed
        if self.global_pool == "token":
            pe = pe[0, 1:]
        return pe.reshape(*self.patch_embed.grid_size, -1)

    def register_attention_hooks(self, attns: List[AttnInfo]) -> List[RemovableHandle]:
        def pre_hook(
            module, inputs, *, name: str, cls_token: bool, resolution: (int, int)
        ):
            attns.append(
                AttnInfo(
                    module=module,
                    name=name,
                    attn=inputs[0].detach(),
                    extra={"cls_token": cls_token, "resolution": resolution},
                )
            )

        handles = []
        for name, module in self.named_modules():
            if isinstance(module, Block):
                # Use a pre-hook on the dropout that follows softmax
                hook = partial(
                    pre_hook,
                    name=name,
                    cls_token=self.global_pool == "token",
                    resolution=self.patch_embed.grid_size,
                )
                handle = module.attn.attn_drop.register_forward_pre_hook(hook)
                handles.append(handle)
        return handles

    def rollout(
        self,
        attns: List[AttnInfo],
        head_reduction: Union[str, Callable[[Tensor], Tensor]] = "mean",
        adjust_residual=True,
    ) -> Tensor:
        """Self-attn rollout: how much output token(s) attend to input tokens across layers

        Args:
            attns: list of attentions with shape ``[B heads Q K]`` where ``Q==K``
            head_reduction: 'mean', 'max', or a callable that reduces the head dimension
            adjust_residual: bool, whether to add 0.5 for the self connection

        Returns:
            Rollout, shape [B Q K]
        """
        attns = [a.attn for a in attns]
        B, head, Q, K = attns[0].shape
        device = attns[0].device

        # Reduce heads: mean or max
        if head_reduction in {"mean", "max"}:
            head_reduction = Reduce("B h Q K -> B Q K", head_reduction)
        attns = [head_reduction(a) for a in attns]

        # adjust for self-connections
        if adjust_residual:
            eye = torch.eye(Q, K, device=device).expand(B, Q, K)
            attns = [a + eye for a in attns]

        roll = torch.eye(Q, K, device=device).expand(B, Q, K)
        for a in attns:
            roll = torch.bmm(a, roll)

        # Pop out CLS token
        if self.global_pool == "token":
            roll = roll[:, 1:, 1:]
        return roll

    def attn_to_img(
        self,
        attn_info: AttnInfo,
        head_reduction: Union[None, str, Callable[[Tensor], Tensor]] = None,
    ):
        """Sum over queries to highlight the most attended keys."""
        if head_reduction in {"mean", "max"}:
            head_reduction = Reduce("B h Q K -> B Q K", head_reduction)

        K_h, K_w = attn_info.extra["resolution"]
        attn = attn_info.attn
        if attn_info.extra["cls_token"]:
            attn = attn[:, :, 1:, 1:]
        if head_reduction is not None:
            attn = head_reduction(attn)
            attn = reduce(attn, "B Q (K_h K_w) -> B K_h K_w", "sum", K_h=K_h)
        else:
            attn = reduce(attn, "B head Q (K_h K_w) -> B head K_h K_w", "sum", K_h=K_h)
        return attn


class MyPatchEmbed(PatchEmbed):
    """Same as timm.PatchEmbed but don't complain about sizes and always flatten."""

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            flatten=True,
        )

    def forward(self, x):
        """Forward.

        Args:
            x: Spatial features ``[B, C, H*P, W*P]``

        Returns:
            Tensor ``[B, H, W, C]``
        """
        x = self.proj(x)
        x = torch.moveaxis(x, -3, -1)
        x = self.norm(x)
        return x
