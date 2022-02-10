from contextlib import contextmanager
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from .attentions import SelfAttentionBlock
from .slot_attention import SlotAttention


class VitGlobalSlotModel(nn.Module):
    def __init__(self, *, backbone, global_fn, obj_fn, global_proj, obj_proj):
        super(VitGlobalSlotModel, self).__init__()
        self.backbone = backbone
        self.global_fn = global_fn
        self.obj_fn = obj_fn
        self.global_proj = global_proj
        self.obj_proj = obj_proj

    def forward(self, images):
        # B 3 H W -> B N C
        f_backbone = self.backbone(images)

        # B N C -> B C
        f_global = self.global_fn(f_backbone)
        p_global = self.global_proj(f_global)

        # B N C -> B K C
        f_slots = self.obj_fn(f_backbone)
        p_slots = self.obj_proj(f_slots)

        return f_backbone, f_global, f_slots, p_global, p_slots


class VitSlotGlobalModel(nn.Module):
    def __init__(self, *, backbone, global_fn, obj_fn, global_proj, obj_proj):
        super(VitSlotGlobalModel, self).__init__()
        self.backbone = backbone
        self.global_fn = global_fn
        self.obj_fn = obj_fn
        self.global_proj = global_proj
        self.obj_proj = obj_proj

    def forward(self, images):
        # B 3 H W -> B N C
        f_backbone = self.backbone(images)

        # B N C -> B K C
        f_slots = self.obj_fn(f_backbone)
        p_slots = self.obj_proj(f_slots)

        # B K C -> B C
        f_global = self.global_fn(f_slots)
        p_global = self.global_proj(f_global)

        return f_backbone, f_global, f_slots, p_global, p_slots


@contextmanager
def vit_slot_forward_with_attns(model: Union[VitGlobalSlotModel, VitSlotGlobalModel]):
    module_to_name: Dict[nn.Module, str] = {}
    vit_attns: Dict[str, torch.Tensor] = {
        # module_name: attn tensor [(A B) heads Q K]
    }
    slot_attns: Dict[str, torch.Tensor] = {
        # iteration_id: attn tensor [(A B) slot K]
    }

    def vit_hook(m, inputs):
        (attns,) = inputs
        vit_attns[module_to_name[m]] = attns.detach()

    def slot_hook(m, inputs):
        attns, iter_idx = inputs
        slot_attns[f"{module_to_name[m]}.{iter_idx}"] = attns.detach()

    def sort_dict(d):
        return {k: d[k] for k in sorted(d.keys())}

    handles: List[RemovableHandle] = []
    for name, module in model.named_modules():
        if isinstance(module, SelfAttentionBlock):
            module_to_name[module.attn.attn_drop] = name
            handle = module.attn.attn_drop.register_forward_pre_hook(vit_hook)
            handles.append(handle)
        elif isinstance(module, SlotAttention):
            module_to_name[module.dot_prod_softmax] = name
            handle = module.dot_prod_softmax.register_forward_pre_hook(slot_hook)
            handles.append(handle)

    def wrapped_forward(*inputs):
        outputs = model(*inputs)
        return outputs, sort_dict(vit_attns), sort_dict(slot_attns)

    try:
        yield wrapped_forward
    finally:
        for handle in handles:
            handle.remove()
