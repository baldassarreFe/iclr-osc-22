import timm.models
from torch import nn as nn


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
        depth=4,
        block_drop: float = 0.0,
        block_attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        # [B 3 H W] -> [B N C], where N = H * W / (patch_size**2)
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
                timm.models.vision_transformer.Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=False,
                    drop=block_drop,
                    attn_drop=block_attn_drop,
                    drop_path=drop_path,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(depth)
            ]
        )

        self.init_weights()

    def init_weights(self):
        # noinspection PyProtectedMember
        self.apply(timm.models.vision_transformer._init_vit_weights)

    def forward(self, images):
        x = self.patch_emb(images)
        for i, block in enumerate(self.attn_blocks):
            if i == 0 or self.pos_embed_every_layer:
                x = self.pos_embed(x)
            x = block(x)
        # output is not layer-normed
        return x
