from __future__ import annotations
import numpy as np
from numpy import ndarray
from typing import Callable, Optional, Type, Union
from typing import SupportsFloat as Numeric
from .grad_fn import *


Value = Union[Numeric, 'Tensor']

def ndfy(some: Union[Value, ndarray]) -> ndarray:
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
        arr: Union[Numeric, ndarray, Tensor],
        requires_grad: bool = False,
        is_leaf: bool = True,
        grad_fn: Optional[GradFn] = None
    ) -> None:
        self.arr = ndfy(arr).copy()
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf
        self.grad_fn = grad_fn
        self.grad: Optional[ndarray] = None
        self.grad_cnt: Optional[int] = 0 if requires_grad else None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.arr.shape

    @property
    def size(self) -> int:
        return self.arr.size

    @property
    def ndim(self) -> int:
        return self.arr.ndim

    def item(self) -> Numeric:
        return self.arr.item()

    def _create_new_tensor(
        self,
        o: Value,
        operation: Callable[[ndarray, ndarray], ndarray],
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

    def backward(self) -> None:
        """
        Performs backpropagation by computing the gradient of the tensor.

        Returns:
            None
        """
        assert self.grad_fn is not None

        self.grad = np.ones_like(self.arr)
        self.grad_fn(self)


    """
    Concept of magic methods:
    In Python, magic methods are special methods that are invoked by the interpreter to perform basic operations.
    For example, the __add__ method is invoked when the + operator is used on two objects.

    For more information, see https://rszalski.github.io/magicmethods/
    """
    def __add__(self, o: Value) -> Tensor:
        """
        Adds two tensors element-wise.

        Args:
            o (Value): The tensor to be added.

        Returns:
            Tensor: A new tensor containing the element-wise sum of the two tensors.
        """
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

    def _assert_not_leaf(self) -> None:
        assert not self.requires_grad
        # assert not self.is_leaf or not self.requires_grad

    def __iadd__(self, o: Value) -> Tensor:
        self._assert_not_leaf()
        self.arr = self.arr + ndfy(o)
        return self

    def __isub__(self, o: Value) -> Tensor:
        self._assert_not_leaf()
        self.arr = self.arr - ndfy(o)
        return self

    def __imul__(self, o: Value) -> Tensor:
        self._assert_not_leaf()
        self.arr = self.arr * ndfy(o)
        return self

    def __itruediv__(self, o: Value) -> Tensor:
        self._assert_not_leaf()
        self.arr = self.arr / ndfy(o)
        return self

    def __ipow__(self, o: Value) -> Tensor:
        self._assert_not_leaf()
        self.arr = self.arr ** ndfy(o)
        return self

    def __imatmul__(self, o: Value) -> Tensor:
        self._assert_not_leaf()
        self.arr = self.arr @ ndfy(o)
        return self

    def __getitem__(self, key) -> Tensor:
        key = key.arr if isinstance(key, Tensor) else key
        return _new_tensor(self, self.arr[key], GetitemGradFn, key=key)

    def __setitem__(self, key, value) -> None:
        self._assert_not_leaf()
        self.arr[key] = value

    def __str__(self) -> str:
        arr = str(self.arr)
        req_grad = ", requires_grad=True" if self.requires_grad else ""
        grad_fn = f", grad_fn={self.grad_fn.__class__.__name__}" if self.grad_fn is not None else ""
        return f"Tensor({arr}{req_grad}{grad_fn})"

    def __repr__(self) -> str:
        return self.__str__()


def _new_tensor(x: Tensor, arr: ndarray, grad_fn: Type[GradFn], **kwargs: Any) -> Tensor:
    if x.requires_grad:
        x.grad_cnt += 1
    return Tensor(
        arr,
        requires_grad=x.requires_grad,
        is_leaf=not x.requires_grad,
        grad_fn=grad_fn(x, **kwargs) if x.requires_grad else None
    )