from __future__ import annotations

import numpy as np
from numpy import ndarray

from typing import Any, Callable, Union, Type

from .tensornode import TensorNode, NodeValue, _new_tensornode
from numpytorch.autograd import GradFn
from numpytorch.autograd._operators import SetitemGradFn, SetitemTensorGradFn


Value = Union[float, 'Tensor']
_Operation = Callable[['Tensor', Value], 'Tensor']

class Tensor:
    def __init__(self, *args, **kwargs) -> None:
        if 'node' in kwargs:
            self.node = kwargs['node']
        else:
            self.node = TensorNode(*args, **kwargs)

    arr_ = property(lambda self: self.node.arr)
    @arr_.setter
    def arr(self, value: ndarray) -> None:
        self.node.arr = value

    requires_grad = property(lambda self: self.node.requires_grad)
    is_leaf = property(lambda self: self.node.is_leaf)
    grad_fn = property(lambda self: self.node.grad_fn)

    grad_ = property(lambda self: self.node.grad)
    @grad_.setter
    def grad(self, value: ndarray | None) -> None:
        self.node.grad = value

    shape = property(lambda self: self.node.shape)
    size = property(lambda self: self.node.size)
    ndim = property(lambda self: self.node.ndim)

    item = lambda self: self.node.item()
    backward = lambda self: self.node.backward()

    @staticmethod
    def _operation(operation:
        Callable[[TensorNode, NodeValue], TensorNode] |
        Callable[[TensorNode, TensorNode], TensorNode]
    ) -> _Operation:
        def new_operation(self: Tensor, o: Value) -> Tensor:
            oo = o.node if isinstance(o, Tensor) else o
            res = operation(self.node, oo)
            return Tensor(node=res)
        return new_operation

    __add__ = _operation(TensorNode.__add__)
    __radd__ = __add__
    __iadd__ = _operation(TensorNode.__iadd__)

    __sub__ = _operation(TensorNode.__sub__)
    __rsub__ = _operation(TensorNode.__rsub__)
    __isub__ = _operation(TensorNode.__isub__)

    __mul__ = _operation(TensorNode.__mul__)
    __rmul__ = _operation(TensorNode.__rmul__)
    __imul__ = _operation(TensorNode.__imul__)

    __truediv__ = _operation(TensorNode.__truediv__)
    __rtruediv__ = _operation(TensorNode.__rtruediv__)
    __itruediv__ = _operation(TensorNode.__itruediv__)

    __pow__ = _operation(TensorNode.__pow__)
    __rpow__ = _operation(TensorNode.__rpow__)
    __ipow__ = _operation(TensorNode.__ipow__)

    __matmul__ = _operation(TensorNode.__matmul__)
    __rmatmul__ = _operation(TensorNode.__rmatmul__)
    __imatmul__ = _operation(TensorNode.__imatmul__)

    __pos__ = lambda self: Tensor(node=self.node.__pos__())
    __neg__ = lambda self: Tensor(node=self.node.__neg__())

    def __str__(self) -> str:
        arr = str(self.node.arr)
        req_grad = ", requires_grad=True" if self.requires_grad else ""
        grad_fn = f", grad_fn={self.grad_fn.__class__.__name__}" if self.grad_fn is not None else ""
        return f"Tensor({arr}{req_grad}{grad_fn})"

    __repr__ = __str__

    __getitem__ = lambda self, key: Tensor(node=self.node.__getitem__(key))

    def __setitem__(self, key, value: Value) -> None:
        if self.requires_grad:
            assert not self.is_leaf

            new_node = TensorNode(self.node, requires_grad=True, is_leaf=False)

            if isinstance(value, Tensor):
                new_node.arr[key] = value.arr
                if value.requires_grad:
                    new_node.grad_fn = SetitemTensorGradFn(self.node, value.node, key)
                    value.node.grad_cnt += 1
                else:
                    new_node.grad_fn = SetitemGradFn(self.node, key)
            else:
                new_node.arr[key] = value
                new_node.grad_fn = SetitemGradFn(self.node, key)

            self.node.grad_cnt += 1
            self.node = new_node

        else:
            if isinstance(value, Tensor):
                self.node.arr[key] = value.node.arr
            else:
                self.node.arr[key] = value


def _new_tensor(x: Tensor, arr: ndarray, grad_fn: Type[GradFn], **kwargs) -> Tensor:
    if x.requires_grad:
        x.node.grad_cnt += 1

    new_node = TensorNode(
        arr,
        requires_grad=x.requires_grad,
        is_leaf=not x.requires_grad,
        grad_fn=grad_fn(x.node, **kwargs) if x.requires_grad else None
    )
    return Tensor(node=new_node)


def _ndfy(some: Value | ndarray) -> ndarray:
    if isinstance(some, Tensor):
        return some.arr
    elif isinstance(some, ndarray):
        return some
    else:
        return np.array(some)