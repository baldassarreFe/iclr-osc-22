"""
Swin utils: record attentions, visualize as images, rollout.
"""

from functools import partial
from typing import Callable, List, Union

from einops import reduce
from einops.layers.torch import Reduce
from timm.models.swin_transformer import SwinTransformer
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from .rollout import AttnInfo


def register_attention_hooks_swin(
    backbone: SwinTransformer, attns: List[AttnInfo]
) -> List[RemovableHandle]:
    from timm.models.swin_transformer import SwinTransformerBlock

    def softmax_hook(
        module,
        inputs,
        output,
        *,
        name: str,
        resolution: (int, int),
        window: int,
        shift: int,
    ):
        attns.append(
            AttnInfo(
                module=module,
                name=name,
                attn=output.detach(),
                extra={"resolution": resolution, "window": window, "shift": shift},
            )
        )

    handles = []
    for name, module in backbone.named_modules():
        if isinstance(module, SwinTransformerBlock):
            # Use a hook on the softmax module, add extra info
            hook = partial(
                softmax_hook,
                name=name,
                resolution=module.input_resolution,
                window=module.window_size,
                shift=module.shift_size,
            )
            handle = module.attn.softmax.register_forward_hook(hook)
            handles.append(handle)
    return handles


def rollout_swin(
    backbone: SwinTransformer,
    attns: List[AttnInfo],
    head_reduction: Union[str, Callable[[Tensor], Tensor]] = "mean",
    adjust_residual=True,
) -> Tensor:
    raise NotImplementedError()


def attn_to_img_swin(
    attn_info: AttnInfo,
    head_reduction: Union[None, str, Callable[[Tensor], Tensor]] = None,
):
    if head_reduction in {"mean", "max"}:
        head_reduction = Reduce("B h Q K -> B Q K", head_reduction)

    K_h, K_w = attn_info.extra["resolution"]
    attn = attn_info.attn
    if head_reduction is not None:
        attn = head_reduction(attn)
        attn = reduce(attn, "B Q (K_h K_w) -> B K_h K_w", reduction="mean", K_h=K_h)
    else:
        attn = reduce(
            attn, "B head Q (K_h K_w) -> B head K_h K_w", reduction="mean", K_h=K_h
        )
    return attn
