from __future__ import annotations
import numpy as np
from numpy import ndarray
from typing import Callable, List, Optional, Union
from typing import SupportsFloat as Numeric
from .grad_fn import *


Value = Union[Numeric, ndarray, Tensor]

def ndfy(some: Value) -> Union[Numeric, ndarray]:
    if isinstance(some, Tensor):
        return some.arr
    else:
        return some

class Tensor:
    def __init__(
        self,
        arr: Union[Numeric, List, ndarray, Tensor],
        requires_grad: bool = False,
        is_leaf: bool = True,
        grad_fn: Optional[Callable] = None
    ) -> None:
        if isinstance(arr, Tensor):
            self.arr = arr.arr
        elif isinstance(arr, ndarray):
            self.arr = arr
        else:
            self.arr = np.array(arr)
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf
        self.grad_fn = grad_fn
        self.grad: Optional[ndarray] = None

    def shape(self) -> Tuple[int, ...]:
        return self.arr.shape

    def _create_new_tensor(
        self,
        o: Value,
        operation: Callable[[ndarray, ndarray], ndarray],
        grad_fn: GradFn
    ) -> Tensor:
        if not isinstance(o, Tensor):
            o = Tensor(o)

        new_arr = operation(self.arr, o.arr)

        if self.requires_grad or o.requires_grad:
            new_requires_grad = True
            new_is_leaf = False
            new_grad_fn = grad_fn(self, o)
        else:
            new_requires_grad = False
            new_is_leaf = True
            new_grad_fn = None

        new_tensor = Tensor(
            arr=new_arr,
            requires_grad=new_requires_grad,
            is_leaf=new_is_leaf,
            grad_fn=new_grad_fn
        )

        return new_tensor

    def backward(self) -> None:
        self.grad = np.ones_like(self.arr, dtype=float)
        if self.grad is not None:
            self.grad_fn(self.grad)

    def __add__(self, o: Value) -> Tensor:
        return self._create_new_tensor(o, lambda x, y: x+y, AddGradFn)

    def __radd__(self, o: Value) -> Tensor:
        return self.__add__(o)

    def __sub__(self, o: Value) -> Tensor:
        return self._create_new_tensor(o, lambda x, y: x-y, SubGradFn)

    def __rsub__(self, o: Value) -> Tensor:
        return self._create_new_tensor(o, lambda x, y: y-x, RSubGradFn)

    def __mul__(self, o: Value) -> Tensor:
        return self._create_new_tensor(o, lambda x, y: x*y, MulGradFn)

    def __rmul__(self, o: Value) -> Tensor:
        return self.__mul__(o)

    def __truediv__(self, o: Value) -> Tensor:
        return self._create_new_tensor(o, lambda x, y: x/y, DivGradFn)

    def __rtruediv__(self, o: Value) -> Tensor:
        return self._create_new_tensor(o, lambda x, y: y/x, RDivGradFn)

    def __pow__(self, o: Value) -> Tensor:
        return self._create_new_tensor(o, lambda x, y: x**y, PowGradFn)

    def __rpow__(self, o: Value) -> Tensor:
        return self._create_new_tensor(o, lambda x, y: y**x, RPowGradFn)

    def __matmul__(self, o: Tensor) -> Tensor:
        return self._create_new_tensor(o, lambda x, y: x@y, MatmulGradFn)

    def __rmatmul__(self, o: Tensor) -> Tensor:
        return self._create_new_tensor(o, lambda x, y: y@x, RMatmulGradFn)

    def __pos__(self) -> Tensor:
        return self

    def __neg__(self) -> Tensor:
        return self.__rsub__(0)

    def _assert_grad(self) -> None:
        assert not self.requires_grad

    def __iadd__(self, o: Value) -> Tensor:
        self._assert_grad()
        self.arr = self.arr + ndfy(o)
        return self

    def __isub__(self, o: Value) -> Tensor:
        self._assert_grad()
        self.arr = self.arr - ndfy(o)
        return self

    def __imul__(self, o: Value) -> Tensor:
        self._assert_grad()
        self.arr = self.arr * ndfy(o)
        return self

    def __itruediv__(self, o: Value) -> Tensor:
        self._assert_grad()
        self.arr = self.arr / ndfy(o)
        return self

    def __ipow__(self, o: Value) -> Tensor:
        self._assert_grad()
        self.arr = self.arr ** ndfy(o)
        return self

    def __imatmul__(self, o: Value) -> Tensor:
        self._assert_grad()
        self.arr = self.arr @ ndfy(o)
        return self

    def __str__(self) -> str:
        arr = str(self.arr)
        req_grad = ", requires_grad=True" if self.requires_grad else ""
        grad_fn = f", grad_fn={self.grad_fn.__class__.__name__}" if self.grad_fn is not None else ""
        return f"Tensor({arr}{req_grad}{grad_fn})"

    def __repr__(self) -> str:
        return self.__str__()