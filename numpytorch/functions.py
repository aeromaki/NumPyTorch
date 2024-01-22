import numpy as np
from numpy import ndarray
import math
from .tensor import Tensor, Value
from .grad_fn import SumGradFn


def tensor(
    v: Value,
    requires_grad: bool = False
) -> Tensor:
    if isinstance(v, Tensor):
        v = v.arr.copy()
    elif isinstance(v, ndarray):
        pass
    else:
        v = np.array(v)
    return Tensor(v, requires_grad=requires_grad)

def zeros(*args, requires_grad: bool = False, **kwargs) -> Tensor:
    return Tensor(np.zeros(*args, **kwargs), requires_grad)

def ones(*args, requires_grad: bool = False, **kwargs) -> Tensor:
    return Tensor(np.ones(*args, **kwargs), requires_grad)

def rand(*args, requires_grad: bool = False, **kwargs) -> Tensor:
    return Tensor(np.random.rand(*args, **kwargs), requires_grad)

def exp(x: Tensor) -> Tensor:
    return math.e ** x

def sigmoid(x: Tensor) -> Tensor:
    return 1 / (1 + exp(-x))

def sum(x: Tensor) -> Tensor:
    return Tensor(np.sum(x.arr), requires_grad=True, is_leaf=False, grad_fn=SumGradFn(x))

def mean(x: Tensor) -> Tensor:
    return sum(x) / x.arr.size