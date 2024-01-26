from __future__ import annotations
from typing import Any, Callable, List
from .tensor import Tensor, Value
from .functions import *


class Parameter(Tensor):
    """
    To manage the tensors used as parameters in the model separately,
    we created this class that inherits from Tensor.
    """
    def __init__(self, x: Tensor) -> None:
        super().__init__(arr=x, requires_grad=True)

    def _init_weight(*args: int) -> Tensor:
        # He Uniform Initialization
        u = (6 / args[0])**0.5
        return tensor(np.random.uniform(-u, u, size=args))

    def new(*args: int) -> Parameter:
        return Parameter(Parameter._init_weight(*args))

class Module:
    """
    A class for conveniently managing each layer, module, or model of a DNN.
    If you want to create a new layer, you can create a subclass that inherits
    from Module and just implement the forward method.
    """
    def _forward_unimplemented(*args, **kwargs) -> None:
        raise Exception("forward not implemented")
    forward: Callable[..., Any] = _forward_unimplemented

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def parameters(self) -> List[Parameter]:
        """
        In order to optimize a model during training, the values of the parameters inside
        the model must be constantly updated. This is done through the optimizer in optim.py,
        which requires a list of all the parameters (Parameter) a model (or module) has.
        If a Module contains other Modules as attributes, it will also return the parameters
        of those Modules.
        """
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