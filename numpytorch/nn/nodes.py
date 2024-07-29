from typing import Collection

from .base import Module, Parameter
from numpytorch.tensor import Tensor


class Sequential(Module):
    """
    It is often the case that multiple layers need to be applied in succession, each taking
    a single tensor as input and returning a single tensor (e.g. CNN). It's tedious to assign
    each layer an attribute for this process and apply each one directly in the forward, so we
    can wrap it in a simple Module.
    """
    def __init__(self, *args) -> None:
        for i, module in enumerate(args):
            setattr(self, str(i), module)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.__dict__.values():
            x = layer(x)
        return x

class ModuleList(Module, list):
    def __init__(self, modules: Collection[Module]) -> None:
        super().__init__(modules)
        for i, module in enumerate(modules):
            setattr(self, str(i), module)