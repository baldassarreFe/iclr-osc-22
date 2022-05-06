import dataclasses
from typing import Any, Dict, Optional

from torch import Tensor
from torch.nn import Module


@dataclasses.dataclass
class AttnInfo(object):
    module: Module
    name: str
    attn: Tensor
    img: Optional[Tensor] = None
    extra: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        # dataclasses.asdict() makes deep copies of all values, we don't want that
        res = {}
        for f in dataclasses.fields(self):
            v = self.__getattribute__(f.name)
            if v is not None:
                res[f.name] = v
        return res
