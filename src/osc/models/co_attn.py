"""
Co-attention.

Block modules wrap an attention module with layer norm,
feed-forward layers and residual connections.

Decoder module (WIP) is a stack of blocks.
"""

from typing import Callable, List, Tuple, Union

from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from osc.models.rollout import AttnInfo

from .cross_attn import CrossAttentionBlock


class CoAttentionBlock(nn.Module):
    """Co-attention block, both ``a->b`` and ``b->a``.

    Check the :func:`forward` method.
    """

    def __init__(self, *args, **kwargs):
        """Init.

        Args:
            *args: Same as :class:`CrossAttentionBlock`
            **kwargs: Same as :class:`CrossAttentionBlock`
        """
        super().__init__()
        self.cross_a_b = CrossAttentionBlock(*args, **kwargs)
        self.cross_b_a = CrossAttentionBlock(*args, **kwargs)

    def forward(self, a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward.

        Args:
            a: tensor of shape ``[B N C]``
            b: tensor of shape ``[B M C]``

        Returns:
            Tensors with shapes:
            ``[B N C]`` (``a`` attending to ``b``),
            ``[B M C]`` (``b`` attending to ``a``).
        """

        new_a = self.cross_a_b(a, b)
        new_b = self.cross_b_a(b, a)
        return new_a, new_b


class CoAttentionDecoder(nn.Module):
    def register_attention_hooks(self, attns: List[AttnInfo]) -> List[RemovableHandle]:
        raise NotImplementedError()

    @staticmethod
    def rollout(
        attns: List[AttnInfo],
        rollout_a_a: Tensor = None,
        rollout_b_b: Tensor = None,
        head_reduction: Union[str, Callable[[Tensor], Tensor]] = "mean",
    ) -> Tensor:
        raise NotImplementedError()

    @staticmethod
    def attn_to_img(
        attn_info: AttnInfo,
        resolution: (int, int),
        head_reduction: Union[str, Callable[[Tensor], Tensor]] = "mean",
    ) -> Tensor:
        raise NotImplementedError()
