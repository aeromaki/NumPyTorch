from __future__ import annotations
from typing import Any, Callable, List
from .tensor import Tensor
from .functions import *


class Parameter(Tensor):
    def __init__(self, x: Tensor) -> None:
        super().__init__(arr=x, requires_grad=True)

    def new(*args: int) -> Parameter:
        return Parameter(rand(*args))

class Module:
    def _forward_unimplemented(*args, **kwargs) -> None:
        raise Exception("forward not implemented")
    forward: Callable[..., Any] = _forward_unimplemented

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def parameters(self) -> List[Parameter]:
        params: List[Parameter] = []
        for v in self.__dict__.values():
            if isinstance(v, Module):
                params += v.parameters()
            elif isinstance(v, Parameter):
                params.append(v)
        return params


class Linear(Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        self.w = Parameter.new(d_in, d_out)
        self.b = Parameter.new(d_out)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.w + self.b

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)

class Sequential(Module):
    def __init__(self, *args) -> None:
        for i, module in enumerate(args):
            setattr(self, str(i), module)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.__dict__.values():
            x = layer(x)
        return x