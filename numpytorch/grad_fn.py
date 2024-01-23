from __future__ import annotations
import numpy as np
from numpy import ndarray
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any, Callable, Optional, Tuple
)
if TYPE_CHECKING:
    from .tensor import Tensor


class GradFn(ABC):
    def __init__(self, *args: 'Tensor') -> None:
        self.tensors: Tuple['Tensor', ...] = args

    def __call__(self, y: 'Tensor') -> None:
        self.propagate(y)

    @classmethod
    @abstractmethod
    def f_d(cls, *args: 'Tensor') -> Tuple[ndarray, ...]:
        pass

    def propagate(self, y: 'Tensor') -> None:
        grads: Tuple[ndarray, ...] = self.f_d(*self.tensors, y)
        for x, dx in zip(self.tensors, grads):
            if dx.shape != x.shape():
                dx = dx.sum(0)
            if x.requires_grad:
                if x.grad is not None:
                    x.grad += dx
                else:
                    x.grad = dx
                if x.grad_fn is not None:
                    x.grad_fn(x)


class SumGradFn(GradFn):
    def __init__(self, x: 'Tensor') -> None:
        super().__init__(x)

    @classmethod
    def f_d(cls, *args: 'Tensor') -> Tuple[ndarray]:
        x, y = args
        dx = np.ones_like(x.arr) * y.grad
        return (dx,)

class ReLUGradFn(GradFn):
    def __init__(self, x: 'Tensor') -> None:
        super().__init__(x)

    @classmethod
    def f_d(cls, *args: 'Tensor') -> Tuple[ndarray]:
        x, y = args
        assert y.grad is not None

        dx = (x.arr > 0) * y.grad
        return (dx,)

class SigmoidGradFn(GradFn):
    def __init__(self, x: 'Tensor') -> None:
        super().__init__(x)

    @classmethod
    def f_d(cls, *args: 'Tensor') -> Tuple[ndarray]:
        x, y = args
        assert y.grad is not None

        dx = y.arr * (1 - y.arr) * y.grad
        return (dx,)

class TanhGradFn(GradFn):
    def __init__(self, x: 'Tensor') -> None:
        super().__init__(x)

    @classmethod
    def f_d(cls, *args: 'Tensor') -> Tuple[ndarray]:
        x, y = args
        assert y.grad is not None

        dx = (1 - y.arr)**2 * y.grad
        return (dx,)

class AddGradFn(GradFn):
    def __init__(self, x0: 'Tensor', x1: 'Tensor') -> None:
        super().__init__(x0, x1)

    @classmethod
    def f_d(cls, *args: 'Tensor') -> Tuple[ndarray, ndarray]:
        x0, x1, y = args
        assert y.grad is not None

        dx0 = np.ones_like(x0.arr) * y.grad
        dx1 = np.ones_like(x1.arr) * y.grad
        return dx0, dx1

class SubGradFn(GradFn):
    def __init__(self, x0: 'Tensor', x1: 'Tensor') -> None:
        super().__init__(x0, x1)

    @classmethod
    def f_d(cls, *args: 'Tensor') -> Tuple[ndarray, ndarray]:
        x0, x1, y = args
        assert y.grad is not None

        dx0 = np.ones_like(x0.arr) * y.grad
        dx1 = -np.ones_like(x1.arr) * y.grad
        return dx0, dx1

class RSubGradFn(GradFn):
    def __init__(self, x0: 'Tensor', x1: 'Tensor') -> None:
        super().__init__(x0, x1)

    @classmethod
    def f_d(cls, *args: 'Tensor') -> Tuple[ndarray, ndarray]:
        x0, x1, y = args
        assert y.grad is not None

        dx0 = -np.ones_like(x0.arr) * y.grad
        dx1 = np.ones_like(x1.arr) * y.grad
        return dx0, dx1

class MulGradFn(GradFn):
    def __init__(self, x0: 'Tensor', x1: 'Tensor') -> None:
        super().__init__(x0, x1)

    @classmethod
    def f_d(cls, *args: 'Tensor') -> Tuple[ndarray, ndarray]:
        x0, x1, y = args
        assert y.grad is not None

        dx0 = x1.arr * y.grad
        dx1 = x0.arr * y.grad
        return dx0, dx1

class DivGradFn(GradFn):
    def __init__(self, x0: 'Tensor', x1: 'Tensor') -> None:
        super().__init__(x0, x1)

    @classmethod
    def f_d(cls, *args: 'Tensor') -> Tuple[ndarray, ndarray]:
        x0, x1, y = args
        assert y.grad is not None

        dx0 = y.grad / x1.arr
        dx1 = -x0.arr / x1.arr**2 * y.grad
        return dx0, dx1

class RDivGradFn(GradFn):
    def __init__(self, x0: 'Tensor', x1: 'Tensor') -> None:
        super().__init__(x0, x1)

    @classmethod
    def f_d(cls, *args: 'Tensor') -> Tuple[ndarray, ndarray]:
        x0, x1, y = args
        assert y.grad is not None

        dx0 = -x1.arr / x0.arr**2 * y.grad
        dx1 = y.grad / x0.arr
        return dx0, dx1

class PowGradFn(GradFn):
    def __init__(self, x0: 'Tensor', x1: 'Tensor') -> None:
        super().__init__(x0, x1)

    @classmethod
    def f_d(cls, *args: 'Tensor') -> Tuple[ndarray, ndarray]:
        x0, x1, y = args
        assert y.grad is not None

        b = x0.arr**x1.arr * y.grad
        dx0 = b * x1.arr / x0.arr
        dx1 = b * np.log(x0.arr)
        return dx0, dx1

class RPowGradFn(GradFn):
    def __init__(self, x0: 'Tensor', x1: 'Tensor') -> None:
        super().__init__(x0, x1)

    @classmethod
    def f_d(cls, *args: 'Tensor') -> Tuple[ndarray, ndarray]:
        x0, x1, y = args
        assert y.grad is not None

        b = x1.arr**x0.arr * y.grad
        dx0 = b * np.log(x1.arr)
        dx1 = b * x0.arr / x1.arr
        return dx0, dx1

class MatmulGradFn(GradFn):
    def __init__(self, x0: 'Tensor', x1: 'Tensor') -> None:
        super().__init__(x0, x1)

    @classmethod
    def f_d(cls, *args: 'Tensor') -> Tuple[ndarray, ndarray]:
        x0, x1, y = args
        assert y.grad is not None

        dx0 = y.grad @ np.moveaxis(x1.arr, -1, -2)
        dx1 = np.moveaxis(x0.arr, -1, -2) @ y.grad
        return dx0, dx1

class RMatmulGradFn(GradFn):
    def __init__(self, x0: 'Tensor', x1: 'Tensor') -> None:
        super().__init__(x0, x1)

    @classmethod
    def f_d(cls, *args: 'Tensor') -> Tuple[ndarray, ndarray]:
        x0, x1, y = args
        assert y.grad is not None

        dx0 = np.moveaxis(x1.arr, -1, -2) @ y.grad
        dx1 = y.grad @ np.moveaxis(x0.arr, -1, -2)
        return dx0, dx1