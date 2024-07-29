from typing import Collection

from .module import Module, Parameter
from numpytorch.tensor import Tensor


class Sequential(Module):
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