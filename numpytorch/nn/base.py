import numpy as np

from typing import Any, Callable

from numpytorch.tensor import Tensor, Value
from numpytorch.functions import *


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

    @staticmethod
    def new(*args: int) -> Parameter:
        return Parameter(Parameter._init_weight(*args))

    @staticmethod
    def new_scalar() -> Parameter:
        return Parameter(rand())


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

    def parameters(self) -> list[Parameter]:
        """
        In order to optimize a model during training, the values of the parameters inside
        the model must be constantly updated. This is done through the optimizer in optim.py,
        which requires a list of all the parameters (Parameter) a model (or module) has.
        If a Module contains other Modules as attributes, it will also return the parameters
        of those Modules.
        """
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