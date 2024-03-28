from __future__ import annotations
import numpy as np
from numpy import ndarray
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Callable, Optional, Tuple, Union
)
if TYPE_CHECKING:
    from .tensor import Tensor


def clip_eps(x: ndarray, eps: float = 1e-06) -> ndarray:
    return np.sign(x) * np.clip(np.abs(x), a_min=eps, a_max=None)


class GradFn(ABC):
    """
    This class is a callable object (function) that is used during the backpropagation process.
    Every tensor in the computational graph with requires_grad=True, starting with the tensor that
    called Tensor.backward, calls this class stored in Tensor.grad_fn to compute (__call__) the
    gradient of its parent tensors, and then calls their Tensor.grad_fn (if grad_fn is not None).
    (Parent tensors likewise compute the gradients of their parent tensors and call grad_fn, in
    effect repeating this until they reach the parameters of the model that require a gradient
    to update their values via gradient descent).

    Attributes:
        tensors (Tuple['Tensor', ...]): The parent tensors of the tensor with this GradFn as Tensor.grad_fn.
                                        Since this function computes the gradient of its parent tensors, not itself,
                                        we need to store the parent tensors.
    """
    def __init__(self, *args: 'Tensor') -> None:
        """
        All operations between tensors are implemented as magic methods in the Tensor class or functions
        in functions.py, making them traceable on the computation graph. When a tensor-to-tensor operation occurs,
        if any tensor has a double requires_grad=True, a GradFn instance corresponding to the operation is created
        and put into the new tensor's grad_fn (resulting from the operation). Just as parent tensors are used in
        the computation, when creating GradFn for a child tensor, you can include its parent tensors in __init__
        to save them.
        """
        self.tensors: Tuple['Tensor', ...] = args

    def __call__(self, y: 'Tensor') -> None:
        self.propagate(y)

    @abstractmethod
    def f_d(self, *args: 'Tensor') -> Tuple[ndarray, ...]:
        """
        GradFn is an abstract base class that provides a uniform backpropagation process for all backpropagation
        functions. Since how the gradient is actually computed depends on what the corresponding forward operation
        is, f_d, the method for computing the gradient of the parent tensors, must be implemented directly in the
        subclasses (which actually have their corresponding forward operation).

        Args:
            *args (Tensor): Parent tensors and itself (Tensor).
        """
        pass

    @staticmethod
    def _handle_broadcast(x: 'Tensor', dx: ndarray) -> ndarray:
        """
        Since ndarray operations often involve broadcasting, it is sometimes necessary to reverse shape the gradient.

        Args:
            x (Tensor): Parent tensor
            dx (ndarray): The gradient of x computed from f_d. We need to fit the shape of this dx to the shape of x.
        """
        if dx.ndim > x.ndim:
            assert dx.shape[-x.ndim:] == x.shape or x.shape == ()
            dx = dx.reshape(-1, *x.shape).sum(0)
        else:
            assert dx.ndim == x.ndim
            for i, (n_dx, n_x) in enumerate(zip(dx.shape, x.shape)):
                if n_x == 1:
                    dx = dx.sum(i, keepdims=True)
        return dx

    def propagate(self, y: 'Tensor') -> None:
        """
        Backward propagation process. The process is as follows
        1. compute the gradient of the parent tensors with self.f_d.
        2. update grad for parent tensors with requires_grad=True (see video for implementation details)
        3. call grad_fn for those parent tensors that have grad_fn.

        Args:
            y (Tensor): A tensor that has this GradFn as its grad_fn.
                        On the computation graph, it is the child tensor that result from the operation.
        """
        # compute the gradient of the parent tensors with self.f_d
        grads: Tuple[ndarray, ...] = self.f_d(*self.tensors, y)
        for x, dx in zip(self.tensors, grads):
            # for parent tensors with requires_grad=True
            if x.requires_grad:
                if x.shape != dx.shape:
                    dx = self._handle_broadcast(x, dx)

                # update grad
                if x.grad is not None:
                    x.grad += dx
                else:
                    x.grad = dx

                x.grad_cnt -= 1
                # call grad_fn for those parent tensors that have a grad_fn
                if x.grad_fn is not None and x.grad_cnt == 0:
                    x.grad_fn(x)


class SumGradFn(GradFn):
    def __init__(
        self,
        x: 'Tensor',
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False
    ) -> None:
        super().__init__(x)
        self.axis = axis
        self.keepdims = keepdims

    def f_d(self, *args: 'Tensor') -> Tuple[ndarray]:
        x, y = args
        assert y.grad is not None

        if self.axis is not None and not self.keepdims:
            grad = np.expand_dims(y.grad, self.axis)
        else:
            grad = y.grad
        dx = np.ones_like(x.arr) * grad
        return (dx,)

class ReLUGradFn(GradFn):
    def __init__(self, x: 'Tensor') -> None:
        super().__init__(x)

    @staticmethod
    def f_d(*args: 'Tensor') -> Tuple[ndarray]:
        x, y = args
        assert y.grad is not None

        dx = (x.arr > 0) * y.grad
        return (dx,)

class LogGradFn(GradFn):
    def __init__(self, x: 'Tensor') -> None:
        super().__init__(x)

    @staticmethod
    def f_d(*args: 'Tensor') -> Tuple[ndarray]:
        x, y = args
        assert y.grad is not None

        dx = y.grad / clip_eps(x.arr)
        return (dx,)

class SigmoidGradFn(GradFn):
    def __init__(self, x: 'Tensor') -> None:
        super().__init__(x)

    @staticmethod
    def f_d(*args: 'Tensor') -> Tuple[ndarray]:
        x, y = args
        assert y.grad is not None

        dx = y.arr * (1 - y.arr) * y.grad
        return (dx,)

class TanhGradFn(GradFn):
    def __init__(self, x: 'Tensor') -> None:
        super().__init__(x)

    @staticmethod
    def f_d(*args: 'Tensor') -> Tuple[ndarray]:
        x, y = args
        assert y.grad is not None

        dx = (1 - y.arr)**2 * y.grad
        return (dx,)

class ReshapeGradFn(GradFn):
    def __init__(self, x: 'Tensor') -> None:
        super().__init__(x)

    @staticmethod
    def f_d(*args: 'Tensor') -> Tuple[ndarray]:
        x, y = args
        assert y.grad is not None

        dx = y.grad.reshape(x.shape)
        return (dx,)

class AddGradFn(GradFn):
    def __init__(self, x0: 'Tensor', x1: 'Tensor') -> None:
        super().__init__(x0, x1)

    @staticmethod
    def f_d(*args: 'Tensor') -> Tuple[ndarray, ndarray]:
        x0, x1, y = args
        assert y.grad is not None

        dx0 = np.ones_like(x0.arr) * y.grad
        dx1 = np.ones_like(x1.arr) * y.grad
        return dx0, dx1

class SubGradFn(GradFn):
    def __init__(self, x0: 'Tensor', x1: 'Tensor') -> None:
        super().__init__(x0, x1)

    @staticmethod
    def f_d(*args: 'Tensor') -> Tuple[ndarray, ndarray]:
        x0, x1, y = args
        assert y.grad is not None

        dx0 = np.ones_like(x0.arr) * y.grad
        dx1 = -np.ones_like(x1.arr) * y.grad
        return dx0, dx1

class MulGradFn(GradFn):
    def __init__(self, x0: 'Tensor', x1: 'Tensor') -> None:
        super().__init__(x0, x1)

    @staticmethod
    def f_d(*args: 'Tensor') -> Tuple[ndarray, ndarray]:
        x0, x1, y = args
        assert y.grad is not None

        dx0 = x1.arr * y.grad
        dx1 = x0.arr * y.grad
        return dx0, dx1

class DivGradFn(GradFn):
    def __init__(self, x0: 'Tensor', x1: 'Tensor') -> None:
        super().__init__(x0, x1)

    @staticmethod
    def f_d(*args: 'Tensor') -> Tuple[ndarray, ndarray]:
        x0, x1, y = args
        assert y.grad is not None

        dx0 = y.grad / clip_eps(x1.arr)
        dx1 = -x0.arr / clip_eps(x1.arr**2) * y.grad
        return dx0, dx1

class PowGradFn(GradFn):
    def __init__(self, x0: 'Tensor', x1: 'Tensor') -> None:
        super().__init__(x0, x1)

    @staticmethod
    def f_d(*args: 'Tensor') -> Tuple[ndarray, ndarray]:
        x0, x1, y = args
        assert y.grad is not None
        # assert (x0.arr > 0).all()

        b = x0.arr**(x1.arr-1) * y.grad
        dx0 = x1.arr * b
        dx1 = np.log(x0.arr) * x0.arr * b
        return dx0, dx1

class MatmulGradFn(GradFn):
    def __init__(self, x0: 'Tensor', x1: 'Tensor') -> None:
        super().__init__(x0, x1)

    @staticmethod
    def f_d(*args: 'Tensor') -> Tuple[ndarray, ndarray]:
        x0, x1, y = args
        assert y.grad is not None

        dx0 = y.grad @ np.moveaxis(x1.arr, -1, -2)
        dx1 = np.moveaxis(x0.arr, -1, -2) @ y.grad
        return dx0, dx1

class RSubGradFn(SubGradFn):
    def __init__(self, x0: 'Tensor', x1: 'Tensor') -> None:
        super().__init__(x1, x0)

class RDivGradFn(DivGradFn):
    def __init__(self, x0: 'Tensor', x1: 'Tensor') -> None:
        super().__init__(x1, x0)

class RPowGradFn(PowGradFn):
    def __init__(self, x0: 'Tensor', x1: 'Tensor') -> None:
        super().__init__(x1, x0)

class RMatmulGradFn(MatmulGradFn):
    def __init__(self, x0: 'Tensor', x1: 'Tensor') -> None:
        super().__init__(x1, x0)

class GetitemGradFn(GradFn):
    def __init__(self, x: 'Tensor', key) -> None:
        super().__init__(x)
        self.key = key

    def f_d(self, *args: 'Tensor') -> Tuple[ndarray]:
        x, y = args
        assert y.grad is not None

        dx = np.zeros_like(x.arr)
        dx[self.key] = y.grad

        return (dx,)

class RepeatGradFn(GradFn):
    def __init__(self, x: 'Tensor') -> None:
        super().__init__(x)

    @staticmethod
    def f_d(self, *args: 'Tensor') -> Tuple[ndarray]:
        x, y = args
        assert y.grad is not None

        dx = y.grad.reshape(*x.shape, -1).sum(-1)

        return (dx,)

class TransposeGradFn(GradFn):
    def __init__(self, x: 'Tensor', axes: Tuple[int, int]) -> None:
        super().__init__(x)
        self.axes = axes

    def f_d(self, *args: 'Tensor') -> Tuple[ndarray]:
        x, y = args
        assert y.grad is not None

        dx = np.swapaxes(y.grad, *self.axes)

        return (dx,)