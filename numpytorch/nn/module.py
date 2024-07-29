from __future__ import annotations

import numpy as np

from typing import Any, Callable

from numpytorch.tensor import Tensor, Value
from numpytorch.functions import tensor, rand, zeros


class Parameter(Tensor):
    def __init__(self, x: Tensor) -> None:
        super().__init__(arr=x, requires_grad=True)

    def _init_weight(*args: int) -> Tensor:
        # He Uniform Initialization
        u = (6 / args[0])**0.5
        return tensor(np.random.uniform(-u, u, size=args))

    @staticmethod
    def new(*args: int) -> Parameter:
        return Parameter(Parameter._init_weight(*args))

    @staticmethod
    def new_scalar() -> Parameter:
        return Parameter(rand())


class Module:
    def _forward_unimplemented(*args, **kwargs) -> None:
        raise Exception("forward not implemented")
    forward: Callable[..., Any] = _forward_unimplemented

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def parameters(self) -> list[Parameter]:
        params: list[Parameter] = []
        for v in self.__dict__.values():
            if isinstance(v, Module):
                params += v.parameters()
            elif isinstance(v, Parameter):
                params.append(v)
        return params


class Linear(Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = True) -> None:
        self.w = Parameter.new(d_in, d_out)
        self.b: Value = Parameter(zeros(d_out)) if bias else 0

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.w + self.b