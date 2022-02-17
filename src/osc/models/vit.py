from typing import Tuple

import timm.models
import torch
from torch import Tensor
from torch import nn as nn

from osc.models.attentions import SelfAttentionBlock


class ViTBackbone(nn.Module):
    def __init__(
        self,
        *,
        img_size=(128, 128),
        pos_embed=None,
        pos_embed_every_layer=False,
        embed_dim=512,
        patch_size=8,
        num_heads=4,
        num_layers=4,
        block_drop: float = 0.0,
        block_attn_drop: float = 0.0,
        drop_path: float = 0.0,
        mlp_ratio: float = 4.0,
        global_pool: str = "cls"
    ):
        super().__init__()

        # [B 3 H W] -> [B N C]
        # where N = H * W / (patch_size**2)
        self.patch_emb = timm.models.layers.PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            norm_layer=None,
            flatten=True,
        )

        if pos_embed is None or pos_embed == "none":
            pos_embed = nn.Identity()
        self.pos_embed = pos_embed
        self.pos_embed_every_layer = pos_embed_every_layer

        # B N C -> B N C
        self.attn_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=False,
                    drop=block_drop,
                    attn_drop=block_attn_drop,
                    drop_path=drop_path,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(num_layers)
            ]
        )

        if global_pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(embed_dim))
        elif global_pool == "avg":
            self.cls_token = None
        else:
            raise ValueError(global_pool)

        self.init_weights()

    def init_weights(self):
        # noinspection PyProtectedMember
        if self.cls_token is not None:
            timm.models.layers.trunc_normal_(self.cls_token, std=0.02)
        self.apply(timm.models.vision_transformer._init_vit_weights)

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward.

        Args:
            images: tensor of shape ``[B 3 H W]``

        Returns:
            A tuple of global and patch features. Global features have shape ``[B C]``.
            Patch features have shape``[B N C]``, where ``N = HW // patch_area``.
            The output is not L2 normalized.
        """
        B = images.shape[0]
        x = self.patch_emb(images)
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(B, 1, -1)
            x = torch.cat([cls_token, x], dim=1)

        for i, block in enumerate(self.attn_blocks):
            if i == 0 or self.pos_embed_every_layer:
                if self.cls_token is not None:
                    x = torch.cat([x[:, :1, :], self.pos_embed(x[:, 1:, :])], dim=1)
                else:
                    x = self.pos_embed(x)
            x = block(x)

        if self.cls_token is not None:
            f_global = x[:, 0, :]
            f_backbone = x[:, 1:, :]
        else:
            f_global = torch.mean(x, dim=1)
            f_backbone = x
        return f_global, f_backbone
