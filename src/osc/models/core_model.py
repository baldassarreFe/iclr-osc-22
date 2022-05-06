"""
Model definitions and wrappers to get attention maps.
"""
from contextlib import AbstractContextManager
from typing import Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from .obj_queries import (
    LearnedObjectTokens,
    NormalObjectTokens,
    PatchObjectTokens,
    SampledObjectTokens,
)
from .rollout import AttnInfo


class ModelOutput(NamedTuple):
    """Model output.

    Fields:

    - ``f_backbone``: backbone features,  shape ``[B H W C]``
    - ``f_global``:   global features,    shape ``[B C]``
    - ``f_slots``:    object features,    shape ``[B C]``
    - ``p_global``:   global projections, shape ``[B S C]``
    - ``p_slots``:    object projections, shape ``[B S C]``

    """

    f_backbone: Tensor
    f_global: Tensor
    p_global: Tensor
    f_slots: Optional[Tensor] = None
    p_slots: Optional[Tensor] = None

    def reshape_batch(self, *shape: int) -> "ModelOutput":
        return ModelOutput(
            **{
                k: (v.reshape(*shape, *v.shape[1:]) if v is not None else None)
                for k, v in self._asdict().items()
            }
        )


class CoreModel(nn.Module):
    """Core model.

    See :func:``forward`` method.
    """

    def __init__(
        self,
        *,
        backbone,
        global_fn,
        global_proj,
        obj_queries=None,
        obj_fn=None,
        obj_proj=None,
    ):
        super().__init__()

        obj_ = [m is None for m in [obj_queries, obj_fn, obj_proj]]
        if not all(obj_) and any(obj_):
            raise ValueError("The obj_ modules must be all missing or all present.")

        self.backbone = backbone
        self.obj_queries = obj_queries
        self.obj_fn = obj_fn
        self.obj_proj = obj_proj
        self.global_fn = global_fn
        self.global_proj = global_proj

    def forward(self, images: Tensor, num_objects: int = None) -> ModelOutput:
        """Forward.

        Args:
            images: float tensor of shape ``[B 3 H W]``
            num_objects:

        Returns:
            A :class:``ModelOutput`` tuple containing per-patch backbone features,
            global features, global projections, object features, object projections.
        """

        # f_backbone_patch: [B H W C]
        _, f_backbone_patch = self.forward_backbone(images)
        B, *_, C = f_backbone_patch.shape
        f_backbone_flat = f_backbone_patch.reshape(B, -1, C)

        if self.obj_queries is None:
            # f_global, p_global: [B HW C] -> [B C] -> [B C]
            f_global = self.global_fn(f_backbone_flat)
            p_global = self.global_proj(f_global)
            return ModelOutput(
                f_backbone=f_backbone_patch, f_global=f_global, p_global=p_global
            )

        else:
            # q_objs, f_slots, p_slots: [B H W C] -> [B S C] -> [B S C]
            q_objs = self.forward_queries(f_backbone_flat, num_objects)
            f_slots = self.obj_fn(q_objs, f_backbone_patch)
            p_slots = self.obj_proj(f_slots)

            # f_global, p_global: [B S C] -> [B C] -> [B C]
            f_global = self.global_fn(f_slots)
            p_global = self.global_proj(f_global)

            return ModelOutput(
                f_backbone=f_backbone_patch,
                f_global=f_global,
                f_slots=f_slots,
                p_global=p_global,
                p_slots=p_slots,
            )

    def forward_backbone(self, images: Tensor) -> (Tensor, Tensor):
        """
        Returns:
            A tuple of ``(f_backbone_global, f_backbone_patch)``
        """
        if isinstance(self.backbone, SwinTransformer):
            x = self.backbone.forward_features(images)
            B, _, C = x.shape
            f_backbone_global = x.mean(-2)
            H, W = self.backbone.layers[-1].input_resolution
            if self.backbone.layers[-1].downsample is not None:
                H, W = H // 2, W // 2
            f_backbone_patch = x.reshape(B, H, W, C)
            return f_backbone_global, f_backbone_patch
        else:
            f_backbone_global, f_backbone_patch = self.backbone(images)

        return f_backbone_global, f_backbone_patch

    def forward_queries(self, f_backbone_flat, num_objects: int = None):
        B = f_backbone_flat.shape[0]
        oq = self.obj_queries
        if isinstance(oq, LearnedObjectTokens):
            if num_objects is not None:
                raise ValueError(f"With {type(oq)} num_objects must be None")
            return oq(B)
        if isinstance(oq, (NormalObjectTokens, SampledObjectTokens)):
            return oq(B, num_objects)
        if isinstance(oq, PatchObjectTokens):
            return oq(f_backbone_flat, num_objects)
        raise NotImplementedError(type(oq))

    @torch.jit.ignore
    def no_weight_decay(self):
        """Recursively check all submodules to find weights to exclude from decay.

        To be used in conjunction with func:`timm.optim.create_optimizer_v2`.
        """
        res = set()
        for name, module in self.named_modules():
            if module is self:
                continue
            if hasattr(module, "no_weight_decay"):
                for nwd in module.no_weight_decay():
                    res.add(f"{name}.{nwd}")
        return res


class RecordAttentions(AbstractContextManager):
    """Context manager to set and remove hooks that collect attention maps.

    Upon entering, returns a dict of lists of :class:`AttnInfo` instances.
    The lists get populated after the forward pass.
    If a module is reused multiple times in the model, it will appear multiple times,
    but always with the name of its first use.
    The lists can be reused for multiple forward passes by calling :func:``list.clear``.

    Example:
        Usage::

        >>> model = CoreModel(...)
        >>> with RecordAttentions(model) as attns:
        >>>     outputs = model(...)
        >>> print(attns)

    """

    def __init__(self, model: CoreModel):
        """Init.

        Args:
            model: Model where the attention hooks will be set.
        """
        self.model = model
        self.handles: Optional[List[RemovableHandle]] = None

    def __enter__(self) -> Dict[str, List[AttnInfo]]:
        if self.handles is not None:
            raise RuntimeError(f"{self.__class__.__name__} is not re-entrant")

        m = self.model
        handles = []
        attns_all = {}

        attns_all["backbone"] = []
        handles.extend(m.backbone.register_attention_hooks(attns_all["backbone"]))
        if self.model.obj_fn is not None:
            attns_all["obj_fn"] = []
            handles.extend(m.obj_fn.register_attention_hooks(attns_all["obj_fn"]))
        attns_all["global_fn"] = []
        handles.extend(m.global_fn.register_attention_hooks(attns_all["global_fn"]))
        self.handles = handles

        return attns_all

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()
