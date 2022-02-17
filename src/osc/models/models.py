"""
Model definitions and wrappers to get attention maps.
"""

from contextlib import contextmanager
from typing import Dict, Generator, List, NamedTuple, Optional

import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from osc.models.attentions import CrossAttention, SelfAttention

from .attentions import SlotAttention
from .embeds import (
    KmeansCosineObjectTokens,
    KmeansEuclideanObjectTokens,
    LearnedObjectTokens,
    SampledObjectTokens,
)


class ModelOutput(NamedTuple):
    """Model output.

    Fields:

    - ``f_backbone``: backbone features,  shape ``[B N C]``
    - ``f_global``:   global features,    shape ``[B C]``
    - ``f_slots``:    object features,    shape ``[B C]``
    - ``p_global``:   global projections, shape ``[B S C]``
    - ``p_slots``:    object projections, shape ``[B S C]``

    """

    f_backbone: Tensor
    f_global: Tensor
    f_slots: Optional[Tensor]
    p_global: Tensor
    p_slots: Optional[Tensor]


class Model(nn.Module):
    """Core model.

    See :func:``forward`` method.
    """

    def __init__(
        self,
        *,
        architecture: str,
        backbone,
        global_fn,
        global_proj,
        obj_queries=None,
        obj_fn=None,
        obj_proj=None,
    ):
        super(Model, self).__init__()
        self.architecture = architecture

        self.backbone = backbone

        self.global_fn = global_fn
        self.global_proj = global_proj

        if architecture == "backbone-global_fn-global_proj":
            if any(m is not None for m in [obj_queries, obj_fn, obj_proj]):
                raise ValueError(
                    f"With {architecture}, all `obj_*` parameters must be None."
                )
        else:
            if any(m is None for m in [obj_queries, obj_fn, obj_proj]):
                raise ValueError(
                    f"With {architecture}, all `obj_*` parameters must not be None."
                )

        self.obj_queries = obj_queries
        self.obj_fn = obj_fn
        self.obj_proj = obj_proj

    def forward(self, images: Tensor) -> ModelOutput:
        """Forward.

        Args:
            images: float tensor of shape ``[B 3 H W]``

        Returns:
            A :class:``ModelOutput`` tuple containing per-patch backbone features,
            global features, global projections, object features, object projections.
        """

        # f_backbone_pool: [B C]
        # f_backbone_patch: [B N C] with N = H*W // num_patches
        f_backbone_pool, f_backbone_patch = self.backbone(images)

        if self.architecture == "backbone-global_fn-global_proj":
            f_global = self.global_fn(f_backbone_pool)
            p_global = self.global_proj(f_global)
            f_slots = None
            p_slots = None
            return ModelOutput(f_backbone_patch, f_global, f_slots, p_global, p_slots)

        if self.architecture == "backbone(-global_fn-global_proj)-obj_fn-obj_proj":
            f_global = self.global_fn(f_backbone_pool)
            p_global = self.global_proj(f_global)

            # q_objs: [B S C]
            q_objs = self._get_q_objs(f_backbone_patch)

            # f_slots, p_slots: [B S C]
            f_slots = self.obj_fn(q_objs, f_backbone_patch)
            p_slots = self.obj_proj(f_slots)

            return ModelOutput(f_backbone_patch, f_global, f_slots, p_global, p_slots)

        if self.architecture == "backbone-obj_fn(-global_fn-global_proj)-obj_proj":
            # q_objs: [B S C]
            q_objs = self._get_q_objs(f_backbone_patch)

            # f_slots, p_slots: [B S C]
            f_slots = self.obj_fn(q_objs, f_backbone_patch)
            p_slots = self.obj_proj(f_slots)

            f_global = self.global_fn(f_slots.mean(dim=1))
            p_global = self.global_proj(f_global)

            return ModelOutput(f_backbone_patch, f_global, f_slots, p_global, p_slots)

        raise ValueError(self.architecture)

    def _get_q_objs(self, f_backbone_patch):
        B, N, C = f_backbone_patch.shape
        if isinstance(self.obj_queries, SampledObjectTokens):
            return self.obj_queries(B)

        if isinstance(self.obj_queries, LearnedObjectTokens):
            return self.obj_queries().expand(B, -1, -1)

        if isinstance(
            self.obj_queries, (KmeansCosineObjectTokens, KmeansEuclideanObjectTokens)
        ):
            return self.obj_queries(f_backbone_patch)

        raise TypeError(type(self.obj_queries))


@contextmanager
def forward_with_attns(model: Model) -> Generator[Dict[str, Tensor], None, None]:
    """Context manager to set and remove hooks that collect attention maps.

    Example:
        Usage::

        >>> with forward_with_attns(model) as attns:
        >>>     outputs = model(inputs)
        >>> print(attns.items())

    Args:
        model: Model where the attention hooks will be set.

    Yields:
        A dictionary that will be populated after the forward pass.
    """
    module_to_name: Dict[nn.Module, str] = {
        # nn.Module: name
    }
    attns: Dict[str, Tensor] = {
        # backbone.attn_blocks.*:          [AB heads Q K]
        # obj_fn.slot_attn.*:              [AB       S K]
        # obj_fn.attn_blocks.*.self_attn:  [AB heads S S]
        # obj_fn.attn_blocks.*.cross_attn: [AB heads S K]
    }

    def normal_hook(m, inputs):
        (a,) = inputs
        attns[module_to_name[m]] = a.detach()

    def slot_hook(m, inputs):
        ((a, iter_idx),) = inputs
        attns[f"{module_to_name[m]}.slot_attn.{iter_idx}"] = a.detach()

    handles: List[RemovableHandle] = []
    for name, module in model.named_modules():
        if isinstance(module, (SelfAttention, CrossAttention)):
            module_to_name[module.attn_drop] = name
            handle = module.attn_drop.register_forward_pre_hook(normal_hook)
            handles.append(handle)

        elif isinstance(module, SlotAttention):
            module_to_name[module.dot_prod_softmax] = name
            handle = module.dot_prod_softmax.register_forward_pre_hook(slot_hook)
            handles.append(handle)

    try:
        yield attns
    finally:
        for handle in handles:
            handle.remove()
