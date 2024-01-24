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
    def __init__(self, *args: 'Tensor') -> None:
        self.tensors: Tuple['Tensor', ...] = args

    def __call__(self, y: 'Tensor') -> None:
        self.propagate(y)

    @abstractmethod
    def f_d(self, *args: 'Tensor') -> Tuple[ndarray, ...]:
        pass

    @staticmethod
    def _handle_broadcast(x: 'Tensor', dx: ndarray) -> ndarray:
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
        grads: Tuple[ndarray, ...] = self.f_d(*self.tensors, y)
        for x, dx in zip(self.tensors, grads):
            if x.requires_grad:
                if x.shape != dx.shape:
                    dx = self._handle_broadcast(x, dx)
                if x.grad is not None and x.grad_fn is None:
                    x.grad += dx
                else:
                    x.grad = dx
                if x.grad_fn is not None:
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
        assert (x0.arr > 0).all()

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