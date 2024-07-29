from __future__ import annotations

import numpy as np
from numpy import ndarray

from typing import (
    Callable, Optional, Type, Union,
    TypeVar, ParamSpec
)

from .grad_fn import *


_P = ParamSpec("_P")
_T = TypeVar("_T")

Value = Union[float, 'Tensor']

_NPOperation = Callable[[ndarray, ndarray], ndarray]
_Operation = Callable[['Tensor', Value], 'Tensor']

def ndfy(some: Value | ndarray) -> ndarray:
    if isinstance(some, Tensor):
        return some.arr
    elif isinstance(some, ndarray):
        return some
    else:
        return np.array(some)

class Tensor:
    """
    A Tensor is a multi-dimensional array used for storing data and performing operations,
    especially in the context of neural networks and computational graphs. This class
    provides an implementation of tensors with automatic differentiation capabilities.

    Attributes:
        arr (ndarray): The underlying data of the tensor stored as a NumPy array.
        requires_grad (bool): If set to True, the tensor will be tracked for gradient computation.
        is_leaf (bool): Indicates whether the tensor is a leaf node in the computation graph.
                        A leaf node is not created from any operation on other tensors.
        grad_fn (Optional[GradFn]): The gradient function associated with
                        the tensor, used to compute gradients during backpropagation.
        grad (Optional[ndarray]): Stores the gradient of the tensor after backpropagation.
    """
    def __init__(
        self,
        arr: float | ndarray | Tensor,
        requires_grad: bool = False,
        is_leaf: bool = True,
        grad_fn: Optional[GradFn] = None
    ) -> None:
        self.arr = ndfy(arr).copy()
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf
        self.grad_fn = grad_fn
        self.grad: Optional[ndarray] = None
        self.grad_cnt = 0

    @property
    def shape(self) -> tuple[int, ...]:
        return self.arr.shape

    @property
    def size(self) -> int:
        return self.arr.size

    @property
    def ndim(self) -> int:
        return self.arr.ndim

    def item(self) -> float:
        return self.arr.item()

    def backward(self) -> None:
        """
        Performs backpropagation by computing the gradient of the tensor.

        Returns:
            None
        """
        assert self.grad_fn is not None
        assert self.shape == ()

        self.grad = np.ones(())
        self.grad_fn(self)

    def _create_new_tensor(
        self,
        o: Value,
        operation: _NPOperation,
        grad_fn: Type[GradFn]
    ) -> Tensor:
        """
        Creates a new tensor by performing an operation on the current tensor and another tensor.
        Also updates:
            - requires_grad: If either of the tensors requires gradient, the new tensor will also require gradient.
            - is_leaf: If neither of the tensors requires gradient, the new tensor will not require gradient.
            - grad_fn: The gradient function for backpropagation.

        Args:
            o (Value): The other tensor to perform the operation with.
            operation (Callable[[ndarray, ndarray], ndarray]): The operation to perform on the tensors.
            grad_fn (GradFn): The gradient function for backpropagation.

        Returns:
            Tensor: The new tensor resulting from the operation.
        """
        if not isinstance(o, Tensor):
            o = Tensor(o)

        new_arr = operation(self.arr, o.arr) # Apply the operation on the arrays

        # If either of the tensors requires gradient, the new tensor will also require gradient and will not be a leaf.
        if self.requires_grad or o.requires_grad:
            new_requires_grad = True
            new_is_leaf = False
            new_grad_fn = grad_fn(self, o)

            if self.requires_grad:
                self.grad_cnt += 1
            if o.requires_grad:
                o.grad_cnt += 1

        else:
            # If neither of the tensors requires gradient, the new tensor will not require gradient and will be a leaf.
            new_requires_grad = False
            new_is_leaf = True
            new_grad_fn = None

        # Create the new tensor with the result of the operation
        new_tensor = Tensor(
            arr=new_arr,
            requires_grad=new_requires_grad,
            is_leaf=new_is_leaf,
            grad_fn=new_grad_fn
        )

        return new_tensor

    @staticmethod
    def _operation(grad_fn: Type[GradFn], operation: _NPOperation) -> _Operation:
        def new_operation(self: Tensor, o: Value) -> Tensor:
            return self._create_new_tensor(o, operation, grad_fn)
        return new_operation

    __add__ = _operation(AddGradFn, lambda x, y: x + y)
    __radd__ = __add__

    __sub__ = _operation(SubGradFn, lambda x, y: x - y)
    __rsub__ = _operation(RSubGradFn, lambda x, y: y - x)

    __mul__ = _operation(MulGradFn, lambda x, y: x * y)
    __rmul__ = __mul__

    __truediv__ = _operation(DivGradFn, lambda x, y: x / y)
    __rtruediv__ = _operation(RDivGradFn, lambda x, y: y / x)

    __pow__ = _operation(PowGradFn, lambda x, y: x ** y)
    __rpow__ = _operation(RPowGradFn, lambda x, y: y ** x)

    __matmul__ = _operation(MatmulGradFn, lambda x, y: x @ y)
    __rmatmul__ = _operation(RMatmulGradFn, lambda x, y: y @ x)

    def __pos__(self) -> Tensor:
        return self
    def __neg__(self) -> Tensor:
        return 0 - self

    @staticmethod
    def _assert_not_leaf(method: Callable[Concatenate[Tensor, _P], _T]) -> Callable[Concatenate[Tensor, _P], _T]:
        def new_f(self: Tensor, *args: _P.args, **kwargs: _P.kwargs) -> _T:
            if self.requires_grad:
                assert not self.is_leaf
            return method(self, *args, **kwargs)
        return new_f

    @_assert_not_leaf
    def __iadd__(self, o: Value) -> Tensor:
        return self + o

    @_assert_not_leaf
    def __isub__(self, o: Value) -> Tensor:
        return self - o

    @_assert_not_leaf
    def __imul__(self, o: Value) -> Tensor:
        return self * o

    @_assert_not_leaf
    def __itruediv__(self, o: Value) -> Tensor:
        return self / o

    @_assert_not_leaf
    def __ipow__(self, o: Value) -> Tensor:
        return self ** o

    @_assert_not_leaf
    def __imatmul__(self, o: Tensor) -> Tensor:
        return self @ o

    def __getitem__(self, key) -> Tensor:
        key = key.arr if isinstance(key, Tensor) else key
        return _new_tensor(self, self.arr[key], GetitemGradFn, key=key)

    @_assert_not_leaf
    def __setitem__(self, key, value: Value) -> None:
        if isinstance(value, Tensor):
            self.arr[key] = value.arr
        else:
            self.arr[key] = value

        if self.grad_fn is not None:
            past_self = Tensor(self, requires_grad=True, grad_fn=self.grad_fn)
            self.grad_cnt += 1
            if isinstance(value, Tensor) and value.requires_grad:
                self.grad_fn = SetitemTensorGradFn(value, key, self.grad_fn)
            else:
                self.grad_fn = SetitemGradFn(key, self.grad_fn)

    def __str__(self) -> str:
        arr = str(self.arr)
        req_grad = ", requires_grad=True" if self.requires_grad else ""
        grad_fn = f", grad_fn={self.grad_fn.__class__.__name__}" if self.grad_fn is not None else ""
        return f"Tensor({arr}{req_grad}{grad_fn})"

    def __repr__(self) -> str:
        return self.__str__()


def _new_tensor(x: Tensor, arr: ndarray, grad_fn: Type[GradFn], **kwargs) -> Tensor:
    if x.requires_grad:
        x.grad_cnt += 1
    return Tensor(
        arr,
        requires_grad=x.requires_grad,
        is_leaf=not x.requires_grad,
        grad_fn=grad_fn(x, **kwargs) if x.requires_grad else None
    )