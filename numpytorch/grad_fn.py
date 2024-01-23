import numpy as np
from numpy import ndarray
from typing import Any, Callable, Optional, Tuple


Tensor = "Tensor"

class GradFn:
    def __init__(
        self,
        f_d: Callable[[Tensor, ..., Tensor], Tuple[ndarray, ...]],
        *args: Tensor
    ) -> None:
        self.tensors = args
        self.f_d = f_d

    def __call__(self, y: Tensor) -> None:
        self.propagate(y)

    def propagate(self, y: Tensor) -> None:
        grads = self.f_d(*self.tensors, y)
        for x, dx in zip(self.tensors, grads):
            if dx.shape != x.arr.shape:
                dx = dx.sum(0)
            if x.requires_grad:
                if x.grad is not None:
                    x.grad += dx
                else:
                    x.grad = dx
                if x.grad_fn is not None:
                    x.grad_fn(x)

class SumGradFn(GradFn):
    def __init__(self, x: Tensor) -> None:
        super().__init__(SumGradFn.f_d, x)

    @staticmethod
    def f_d(x: Tensor, y: Tensor) -> (ndarray,):
        dx = np.ones_like(x.arr) * y.grad
        return (dx,)

class ReLUGradFn(GradFn):
    def __init__(self, x: Tensor) -> None:
        super().__init__(ReLUGradFn.f_d, x)

    @staticmethod
    def f_d(x: Tensor, y: Tensor) -> (ndarray,):
        dx = (x.arr > 0) * y.grad
        return (dx,)

class SigmoidGradFn(GradFn):
    def __init__(self, x: Tensor) -> None:
        super().__init__(SigmoidGradFn.f_d, x)

    @staticmethod
    def f_d(x: Tensor, y: Tensor) -> (ndarray,):
        dx = y.arr * (1 - y.arr) * y.grad
        return (dx,)

class TanhGradFn(GradFn):
    def __init__(self, x: Tensor) -> None:
        super().__init__(TanhGradFn.f_d, x)

    @staticmethod
    def f_d(x: Tensor, y: Tensor) -> (ndarray,):
        dx = (1 - y.arr)**2 * y.grad
        return (dx,)

class AddGradFn(GradFn):
    def __init__(self, x0: Tensor, x1: Tensor) -> None:
        super().__init__(AddGradFn.f_d, x0, x1)

    @staticmethod
    def f_d(x0: Tensor, x1: Tensor, y: Tensor) -> (ndarray, ndarray):
        dx0 = np.ones_like(x0.arr) * y.grad
        dx1 = np.ones_like(x1.arr) * y.grad
        return dx0, dx1

class SubGradFn(GradFn):
    def __init__(self, x0: Tensor, x1: Tensor) -> None:
        super().__init__(SubGradFn.f_d, x0, x1)

    @staticmethod
    def f_d(x0: Tensor, x1: Tensor, y: Tensor) -> (ndarray, ndarray):
        dx0 = np.ones_like(x0.arr) * y.grad
        dx1 = -np.ones_like(x1.arr) * y.grad
        return dx0, dx1

class RSubGradFn(GradFn):
    def __init__(self, x0: Tensor, x1: Tensor) -> None:
        super().__init__(RSubGradFn.f_d, x0, x1)

    @staticmethod
    def f_d(x0: Tensor, x1: Tensor, y: Tensor) -> (ndarray, ndarray):
        dx0 = -np.ones_like(x0.arr) * y.grad
        dx1 = np.ones_like(x1.arr) * y.grad
        return dx0, dx1

class MulGradFn(GradFn):
    def __init__(self, x0: Tensor, x1: Tensor) -> None:
        super().__init__(MulGradFn.f_d, x0, x1)

    @staticmethod
    def f_d(x0: Tensor, x1: Tensor, y: Tensor) -> (ndarray, ndarray):
        dx0 = x1.arr * y.grad
        dx1 = x0.arr * y.grad
        return dx0, dx1

class DivGradFn(GradFn):
    def __init__(self, x0: Tensor, x1: Tensor) -> None:
        super().__init__(DivGradFn.f_d, x0, x1)

    @staticmethod
    def f_d(x0: Tensor, x1: Tensor, y: Tensor) -> (ndarray, ndarray):
        dx0 = y.grad / x1.arr
        dx1 = -x0.arr / x1.arr**2 * y.grad
        return dx0, dx1

class RDivGradFn(GradFn):
    def __init__(self, x0: Tensor, x1: Tensor) -> None:
        super().__init__(RDivGradFn.f_d, x0, x1)

    @staticmethod
    def f_d(x0: Tensor, x1: Tensor, y: Tensor) -> (ndarray, ndarray):
        dx0 = -x1.arr / x0.arr**2 * y.grad
        dx1 = grad / x0.arr
        return dx0, dx1

class PowGradFn(GradFn):
    def __init__(self, x0: Tensor, x1: Tensor) -> None:
        super().__init__(PowGradFn.f_d, x0, x1)

    @staticmethod
    def f_d(x0: Tensor, x1: Tensor, y: Tensor) -> (ndarray, ndarray):
        b = x0.arr**x1.arr * y.grad
        dx0 = b * x1.arr / x0.arr
        dx1 = b * np.log(x0.arr)
        return dx0, dx1

class RPowGradFn(GradFn):
    def __init__(self, x0: Tensor, x1: Tensor) -> None:
        super().__init__(RPowGradFn.f_d, x0, x1)

    @staticmethod
    def f_d(x0: Tensor, x1: Tensor, y: Tensor) -> (ndarray, ndarray):
        b = x1.arr**x0.arr * y.grad
        dx0 = b * np.log(x1.arr)
        dx1 = b * x0.arr / x1.arr
        return dx0, dx1

class MatmulGradFn(GradFn):
    def __init__(self, x0: Tensor, x1: Tensor) -> None:
        super().__init__(MatmulGradFn.f_d, x0, x1)

    @staticmethod
    def f_d(x0: Tensor, x1: Tensor, y: Tensor) -> (ndarray, ndarray):
        dx0 = y.grad @ np.moveaxis(x1.arr, -1, -2)
        dx1 = np.moveaxis(x0.arr, -1, -2) @ y.grad
        return dx0, dx1

class RMatmulGradFn(GradFn):
    def __init__(self, x0: Tensor, x1: Tensor) -> None:
        super().__init__(RMatmulGradFn.f_d, x0, x1)

    @staticmethod
    def f_d(x0: Tensor, x1: Tensor, y: Tensor) -> (ndarray, ndarray):
        dx0 = np.moveaxis(x1.arr, -1, -2) @ y.grad
        dx1 = y.grad @ np.moveaxis(x0.arr, -1, -2)
        return dx0, dx1