"""
Build model according to the configuration.
"""
import logging

import timm
import torch
import torch.nn
from omegaconf import DictConfig
from timm.models.swin_transformer import SwinTransformer
from torch import Tensor

from .core_model import CoreModel
from .cross_attn import CrossAttentionDecoder, CrossAttentionPooling
from .embeds import create_positional_embedding
from .obj_queries import (
    LearnedObjectTokens,
    NormalObjectTokens,
    PatchObjectTokens,
    SampledObjectTokens,
)
from .slot_attn import SlotAttentionDecoder
from .utils import MLP
from .vit import MyVisionTransformer

log = logging.getLogger(__name__)


def build_core_model(cfg: DictConfig) -> CoreModel:
    """Build core model"""
    backbone = build_backbone(cfg)
    global_fn = build_global_fn(cfg)
    global_proj = MLP(in_features=cfg.model.backbone.embed_dim, **cfg.model.global_proj)
    obj_queries = build_obj_queries(cfg)
    obj_fn = build_obj_fn(cfg, backbone)
    obj_proj = (
        MLP(in_features=cfg.model.backbone.embed_dim, **cfg.model.obj_proj)
        if "obj_proj" in cfg.model
        else None
    )
    return CoreModel(
        backbone=backbone,
        global_fn=global_fn,
        global_proj=global_proj,
        obj_queries=obj_queries,
        obj_fn=obj_fn,
        obj_proj=obj_proj,
    )


def build_backbone(cfg: DictConfig) -> torch.nn.Module:
    if cfg.model.backbone.name == "vit":
        backbone = MyVisionTransformer(
            img_size=tuple(cfg.data.crops.large.size),
            patch_size=cfg.model.backbone.patch_size,
            embed_dim=cfg.model.backbone.embed_dim,
            pos_embed_drop=cfg.model.backbone.pos_embed_drop,
            depth=cfg.model.backbone.num_layers,
            num_heads=cfg.model.backbone.num_heads,
            mlp_ratio=cfg.model.backbone.mlp_ratio,
            qkv_bias=cfg.model.backbone.qkv_bias,
            drop_rate=cfg.model.backbone.proj_drop,
            attn_drop_rate=cfg.model.backbone.attn_drop,
            drop_path_rate=cfg.model.backbone.drop_path,
            output_norm=cfg.model.backbone.output_norm,
        )
    elif cfg.model.backbone.name == "swin":
        backbone = SwinTransformer(
            img_size=224,
            global_pool="avg",
            num_classes=0,
            embed_dim=cfg.model.backbone.embed_dim,
            patch_size=cfg.model.backbone.patch_size,
            depths=cfg.model.backbone.num_layers,
            num_heads=cfg.model.backbone.num_heads,
            window_size=cfg.model.backbone.window_size,
            mlp_ratio=cfg.model.backbone.mlp_ratio,
            qkv_bias=cfg.model.backbone.qkv_bias,
            drop_rate=cfg.model.backbone.proj_drop,
            attn_drop_rate=cfg.model.backbone.attn_drop,
            drop_path_rate=cfg.model.backbone.drop_path,
        )
        # TODO create subclass MySwinTransformer and pass these as params
        backbone.pos_drop.p = cfg.model.backbone.pos_embed_drop
        if not cfg.model.backbone.output_norm:
            backbone.norm = torch.nn.Identity()
    else:
        raise NotImplementedError(cfg.mode.backbone.name)

    if cfg.model.backbone.pretrained is not None:
        pre = timm.create_model(cfg.model.backbone.pretrained, pretrained=True)
        missing, unexpected = backbone.load_state_dict(pre.state_dict())
        if len(missing) > 0 or len(unexpected) > 0:
            log.warning(
                "Loading pretrained weights.\nUnexpected: %s\nUnexpected: %s",
                str(missing),
                str(unexpected),
            )

    if cfg.model.backbone.frozen:
        for p in backbone.parameters():
            p.requires_grad_(False)

    return backbone


def build_obj_queries(cfg):
    if "obj_queries" not in cfg.model:
        return None

    if cfg.model.obj_queries.name == "sample":
        return SampledObjectTokens(
            embed_dim=cfg.model.backbone.embed_dim,
            num_objects=cfg.model.obj_queries.num_objects,
            num_components=cfg.model.obj_queries.num_components,
        )

    if cfg.model.obj_queries.name == "normal":
        return NormalObjectTokens(
            embed_dim=cfg.model.backbone.embed_dim,
            num_objects=cfg.model.obj_queries.num_objects,
        )

    if cfg.model.obj_queries.name == "patch":
        return PatchObjectTokens(
            num_objects=cfg.model.obj_queries.num_objects,
        )

    if cfg.model.obj_queries.name == "learned":
        return LearnedObjectTokens(
            embed_dim=cfg.model.backbone.embed_dim,
            num_objects=cfg.model.obj_queries.num_objects,
        )

    raise NotImplementedError(cfg.model.obj_queries.name)


def build_obj_fn(cfg, backbone):
    if "obj_fn" not in cfg.model:
        return None

    num_patches = backbone.patch_embed.grid_size

    if cfg.model.obj_fn.pos_embed is None:
        pos_embed = None
    elif cfg.model.obj_fn.pos_embed == "backbone":
        pos_embed = backbone.get_pos_embed().detach().clone()
    elif cfg.model.obj_fn.pos_embed == "learned":
        pos_embed = create_positional_embedding(
            num_patches, cfg.model.backbone.embed_dim, flatten=False
        )
    else:
        raise NotImplementedError(cfg.model.obj_fn.pos_embed)

    if cfg.model.obj_fn.name == "slot-attention":
        return SlotAttentionDecoder(
            dim=cfg.model.backbone.embed_dim,
            pos_embed=pos_embed,
            pos_embed_drop=cfg.model.obj_fn.pos_embed_drop,
            num_layers=cfg.model.obj_fn.num_layers,
            mlp_ratio=cfg.model.obj_fn.mlp_ratio,
            mlp_drop=cfg.model.obj_fn.mlp_drop,
            fixed_point=cfg.model.obj_fn.fixed_point,
            bias_first=cfg.model.obj_fn.bias_first,
        )

    if cfg.model.obj_fn.name == "cross-attention":
        return CrossAttentionDecoder(
            dim=cfg.model.backbone.embed_dim,
            pos_embed=pos_embed,
            pos_embed_drop=cfg.model.obj_fn.pos_embed_drop,
            reuse_layers=cfg.model.obj_fn.reuse_layers,
            num_layers=cfg.model.obj_fn.num_layers,
            num_heads=cfg.model.obj_fn.num_heads,
            proj_drop=cfg.model.obj_fn.proj_drop,
            attn_drop=cfg.model.obj_fn.attn_drop,
            drop_path=cfg.model.obj_fn.drop_path,
            mlp_ratio=cfg.model.obj_fn.mlp_ratio,
            mlp_drop=cfg.model.obj_fn.mlp_drop,
            qkv_bias=cfg.model.obj_fn.qkv_bias,
        )

    raise NotImplementedError(cfg.model.obj_fn.name)


def get_pos_embed_backbone(backbone: torch.nn.Module, as_grid: bool) -> Tensor:
    if hasattr(backbone, "get_pos_embed"):
        return backbone.get_pos_embed(as_grid=as_grid)
    if isinstance(backbone, SwinTransformer):
        pe = backbone.absolute_pos_embed.squeeze(0)
        if as_grid:
            pe = pe.reshape(*backbone.patch_embed.grid_size, -1)
        return pe
    raise NotImplementedError(type(backbone))


def build_global_fn(cfg: DictConfig):
    if cfg.model.global_fn.name == "cross-attention-pool":
        return CrossAttentionPooling(
            dim=cfg.model.backbone.embed_dim,
            reuse_layers=cfg.model.global_fn.reuse_layers,
            num_layers=cfg.model.global_fn.num_layers,
            num_heads=cfg.model.global_fn.num_heads,
            proj_drop=cfg.model.global_fn.proj_drop,
            attn_drop=cfg.model.global_fn.attn_drop,
            drop_path=cfg.model.global_fn.drop_path,
            mlp_ratio=cfg.model.global_fn.mlp_ratio,
            mlp_drop=cfg.model.global_fn.mlp_drop,
            qkv_bias=cfg.model.global_fn.qkv_bias,
        )
    raise NotImplementedError(cfg.model.global_fn.name)
