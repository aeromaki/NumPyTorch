from __future__ import annotations
from typing import Any, Callable, List
from .tensor import Tensor, Value
from .functions import *


class Parameter(Tensor):
    def __init__(self, x: Tensor) -> None:
        super().__init__(arr=x, requires_grad=True)

    def _init_weight(*args: int) -> Tensor:
        u = (6 / args[0])**0.5
        return tensor(np.random.uniform(-u, u, size=args))

    def new(*args: int) -> Parameter:
        return Parameter(Parameter._init_weight(*args))

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
    def __init__(self, d_in: int, d_out: int, bias: bool = True) -> None:
        self.w = Parameter.new(d_in, d_out)
        self.b: Value = Parameter(zeros(d_out)) if bias else 0

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.w + self.b

class Sequential(Module):
    def __init__(self, *args) -> None:
        for i, module in enumerate(args):
            setattr(self, str(i), module)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.__dict__.values():
            x = layer(x)
        return x

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return tanh(x)

class CrossEntropyLoss(Module):
    def forward(self, logits: Tensor, q: Tensor) -> Tensor:
        if logits.shape != q.shape:
            q = one_hot(q, logits.shape[-1])
        log_p = logits - log(sum(exp(logits), -1, keepdims=True))
        ce = -sum(q * log_p, -1)
        return mean(ce)