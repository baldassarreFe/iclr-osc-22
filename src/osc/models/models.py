"""
Model definitions and wrappers to get attention maps.
"""

from contextlib import contextmanager
from typing import Dict, Generator, List, NamedTuple

import torch
import torch.nn as nn
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

    - f_backbone: backbone features, shape ``[B N C]``
    - f_global: global features, shape ``[B C]``
    - f_slots: object features, shape ``[B C]``
    - p_global: global projections, shape ``[B S C]``
    - p_slots: object projections shape ``[B S C]``

    """

    f_backbone: torch.Tensor
    f_global: torch.Tensor
    f_slots: torch.Tensor
    p_global: torch.Tensor
    p_slots: torch.Tensor


class Model(nn.Module):
    def __init__(
        self,
        *,
        architecture: str,
        backbone,
        global_fn,
        global_proj,
        obj_queries,
        obj_fn,
        obj_proj,
    ):
        super(Model, self).__init__()
        self.architecture = architecture

        self.backbone = backbone

        self.global_fn = global_fn
        self.global_proj = global_proj

        self.obj_queries = obj_queries
        self.obj_fn = obj_fn
        self.obj_proj = obj_proj

    def forward(self, images: torch.Tensor) -> ModelOutput:
        # images: B 3 H W
        B = images.shape[0]

        # f_global: [B C]
        # f_backbone: [B N C]
        f_global, f_backbone = self.backbone(images)

        # q_objs: [B S C]
        if isinstance(self.obj_queries, SampledObjectTokens):
            q_objs = self.obj_queries(B)
        elif isinstance(self.obj_queries, LearnedObjectTokens):
            q_objs = self.obj_queries().expand(B, -1, -1)
        elif isinstance(
            self.obj_queries, (KmeansCosineObjectTokens, KmeansEuclideanObjectTokens)
        ):
            q_objs = self.obj_queries.forward(f_backbone)
        else:
            raise TypeError(type(self.obj_queries))

        # f_slots, p_slots: [B S C]
        f_slots = self.obj_fn(q_objs, f_backbone)
        p_slots = self.obj_proj(f_slots)

        # f_global, p_global: [B C]
        if self.architecture == "backbone(-global_fn-global_proj)-obj_fn-obj_proj":
            f_global = f_global
        elif self.architecture == "backbone-obj_fn(-global_fn-global_proj)-obj_proj":
            f_global = f_slots.mean(dim=1)
        else:
            raise ValueError(self.architecture)
        f_global = self.global_fn(f_global)
        p_global = self.global_proj(f_global)

        return ModelOutput(f_backbone, f_global, f_slots, p_global, p_slots)


@contextmanager
def forward_with_attns(model: Model) -> Generator[Dict[str, torch.Tensor], None, None]:
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
    attns: Dict[str, torch.Tensor] = {
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
