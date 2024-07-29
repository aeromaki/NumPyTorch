import numpy as np
from numpy import ndarray

import math
from typing import Any, Optional, Type

from numpytorch.tensor import _ndfy, _new_tensor
from numpytorch.tensor import *
from numpytorch.autograd._functions import *


def tensor(
    v: Value | ndarray,
    requires_grad: bool = False
) -> Tensor:
    v = _ndfy(v).copy()
    return Tensor(v, requires_grad=requires_grad)

def zeros(*args, requires_grad: bool = False, **kwargs) -> Tensor:
    return Tensor(np.zeros(args, **kwargs), requires_grad)

def ones(*args, requires_grad: bool = False, **kwargs) -> Tensor:
    return Tensor(np.ones(args, **kwargs), requires_grad)

def rand(*args, requires_grad: bool = False, **kwargs) -> Tensor:
    return Tensor(np.random.rand(*args, **kwargs), requires_grad)

def exp(x: Tensor) -> Tensor:
    return math.e ** x

def sigmoid_naive(x: Tensor) -> Tensor:
    return 1 / (1 + exp(-x))

def log(x: Tensor) -> Tensor:
    return _new_tensor(x, np.log(x.arr), LogGradFn)

def sigmoid(x: Tensor) -> Tensor:
    return _new_tensor(x, 1 / (1 + np.exp(-x.arr)), SigmoidGradFn)

def sum(
    x: Tensor,
    axis: Optional[int | tuple[int, ...]] = None,
    keepdims: bool = False
) -> Tensor:
    return _new_tensor(x, np.sum(x.arr, axis, keepdims=keepdims), SumGradFn,
                        axis=axis, keepdims=keepdims)

def mean(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    if axis is None:
        return sum(x) / x.size
    else:
        return sum(x, axis, keepdims) / x.shape[axis]

def var(x: Tensor, axis: int) -> Tensor:
    return mean(x**2, axis, keepdims=True) - mean(x, axis, keepdims=True)**2

def relu(x: Tensor) -> Tensor:
    return _new_tensor(x, np.maximum(0, x.arr), ReLUGradFn)

def tanh(x: Tensor) -> Tensor:
    return _new_tensor(x, np.tanh(x.arr), TanhGradFn)

def reshape(x: Tensor, shape: tuple[int, ...]) -> Tensor:
    return _new_tensor(x, x.arr.reshape(shape), ReshapeGradFn)

def one_hot(x: Tensor, n_label: int) -> Tensor:
    return tensor(np.eye(n_label)[x.arr])

def repeat(x: Tensor, rep: tuple[int, ...]) -> Tensor:
    return _new_tensor(x, np.tile(x.arr, rep), RepeatGradFn)

def unsqueeze(x: Tensor, axis: int) -> Tensor:
    return reshape(x, (*x.shape[:axis], 1, *x.shape[axis:]))

def squeeze(x: Tensor, axis: int) -> Tensor:
    if x.shape[axis] == 1:
        return reshape(x, (*x.shape[:axis], *x.shape[axis+1:]))
    else:
        return x

def transpose(x: Tensor, axes: Optional[tuple[int, int]] = None) -> Tensor:
    if axes is None:
        axes = (-1, -2)
    return _new_tensor(x, np.swapaxes(x.arr, *axes), TransposeGradFn, axes=axes)

def softmax(x: Tensor, axis: int = -1) -> Tensor:
    e = exp(x)
    return e / sum(e, axis, keepdims=True)