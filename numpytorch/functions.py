import numpy as np
from numpy import ndarray
import math
from typing import Tuple, Type
from .tensor import *
from .grad_fn import *


def tensor(
    v: Value,
    requires_grad: bool = False
) -> Tensor:
    v = ndfy(v).copy()
    return Tensor(v, requires_grad=requires_grad)

def zeros(*args, requires_grad: bool = False, **kwargs) -> Tensor:
    return Tensor(np.zeros(*args, **kwargs), requires_grad)

def ones(*args, requires_grad: bool = False, **kwargs) -> Tensor:
    return Tensor(np.ones(*args, **kwargs), requires_grad)

def rand(*args, requires_grad: bool = False, **kwargs) -> Tensor:
    return Tensor(np.random.rand(*args, **kwargs), requires_grad)

def exp(x: Tensor) -> Tensor:
    return math.e ** x

def sigmoid_naive(x: Tensor) -> Tensor:
    return 1 / (1 + exp(-x))

def _new_tensor(x: Tensor, arr: ndarray, grad_fn: Type[GradFn]) -> Tensor:
    return Tensor(
        arr,
        requires_grad=x.requires_grad,
        is_leaf=not x.requires_grad,
        grad_fn=grad_fn(x) if x.requires_grad else None
    )

def sigmoid(x: Tensor) -> Tensor:
    return _new_tensor(x, 1 / (1 + np.exp(-x.arr)), SigmoidGradFn)

def sum(x: Tensor) -> Tensor:
    return _new_tensor(x, np.sum(x.arr), SumGradFn)

def mean(x: Tensor) -> Tensor:
    return sum(x) / x.size

def relu(x: Tensor) -> Tensor:
    return _new_tensor(x, np.clip(x.arr, a_min=0, a_max=None), ReLUGradFn)

def tanh(x: Tensor) -> Tensor:
    return _new_tensor(x, np.tanh(x.arr), TanhGradFn)

def reshape(x: Tensor, shape: Tuple[int, ...]) -> Tensor:
    return _new_tensor(x, x.arr.reshape(shape), ReshapeGradFn)