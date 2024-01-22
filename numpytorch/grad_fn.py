import numpy as np
from numpy import ndarray
from typing import Any, Callable, Optional, Tuple


Tensor = "Tensor"

class GradFn:
    def __init__(
        self,
        f_d: Callable[[Tensor, ..., ndarray], Tuple[ndarray, ...]],
        *args: Tensor
    ) -> None:
        self.tensors = args
        self.f_d = f_d

    def __call__(self, grad: ndarray, b: Optional[Tensor] = None) -> None:
        self.propagate(grad, b)

    def propagate(self, grad: ndarray, b: Optional[Tensor] = None) -> None:
        grads = self.f_d(*self.tensors, grad, b)
        for x, dx in zip(self.tensors, grads):
            if dx.shape != x.arr.shape:
                dx = dx.sum(0)
            if x.requires_grad:
                if x.grad is not None:
                    x.grad += dx
                else:
                    x.grad = dx
                if x.grad_fn is not None:
                    x.grad_fn(x.grad, x)

class SumGradFn(GradFn):
    def __init__(self, x: Tensor) -> None:
        super().__init__(SumGradFn.f_d, x)

    @staticmethod
    def f_d(x: Tensor, grad: ndarray, *args, **kwargs) -> (ndarray,):
        d_x = np.ones_like(x.arr) * grad
        return (d_x,)

class ReLUGradFn(GradFn):
    def __init__(self, x: Tensor) -> None:
        super().__init__(ReLUGradFn.f_d, x)

    @staticmethod
    def f_d(x: Tensor, grad: ndarray, *args, **kwargs) -> (ndarray,):
        d_x = (x.arr > 0) * grad
        return (d_x,)

class SigmoidGradFn(GradFn):
    def __init__(self, x: Tensor) -> None:
        super().__init__(SigmoidGradFn.f_d, x)

    @staticmethod
    def f_d(x: Tensor, grad: ndarray, b: Tensor, *args, **kwargs) -> (ndarray,):
        d_x = b.arr * (1 - b.arr) * grad
        return (d_x,)

class TanhGradFn(GradFn):
    def __init__(self, x: Tensor) -> None:
        super().__init__(TanhGradFn.f_d, x)

    @staticmethod
    def f_d(x: Tensor, grad: ndarray, b: Tensor, *args, **kwargs) -> (ndarray,):
        d_x = (1 - b.arr)**2 * grad
        return (d_x,)

class AddGradFn(GradFn):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(AddGradFn.f_d, x, y)

    @staticmethod
    def f_d(x: Tensor, y: Tensor, grad: ndarray, *args, **kwargs) -> (ndarray, ndarray):
        d_x = np.ones_like(x.arr) * grad
        d_y = np.ones_like(y.arr) * grad
        return d_x, d_y

class SubGradFn(GradFn):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(SubGradFn.f_d, x, y)

    @staticmethod
    def f_d(x: Tensor, y: Tensor, grad: ndarray, *args, **kwargs) -> (ndarray, ndarray):
        d_x = np.ones_like(x.arr) * grad
        d_y = -np.ones_like(y.arr) * grad
        return d_x, d_y

class RSubGradFn(GradFn):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(RSubGradFn.f_d, x, y)

    @staticmethod
    def f_d(x: Tensor, y: Tensor, grad: ndarray, *args, **kwargs) -> (ndarray, ndarray):
        d_x = -np.ones_like(x.arr) * grad
        d_y = np.ones_like(y.arr) * grad
        return d_x, d_y

class MulGradFn(GradFn):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(MulGradFn.f_d, x, y)

    @staticmethod
    def f_d(x: Tensor, y: Tensor, grad: ndarray, *args, **kwargs) -> (ndarray, ndarray):
        d_x = y.arr * grad
        d_y = x.arr * grad
        return d_x, d_y

class DivGradFn(GradFn):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(DivGradFn.f_d, x, y)

    @staticmethod
    def f_d(x: Tensor, y: Tensor, grad: ndarray, *args, **kwargs) -> (ndarray, ndarray):
        d_x = grad / y.arr
        d_y = -x.arr / y.arr**2 * grad
        return d_x, d_y

class RDivGradFn(GradFn):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(RDivGradFn.f_d, x, y)

    @staticmethod
    def f_d(x: Tensor, y: Tensor, grad: ndarray, *args, **kwargs) -> (ndarray, ndarray):
        d_x = -y.arr / x.arr**2 * grad
        d_y = grad / x.arr
        return d_x, d_y

class PowGradFn(GradFn):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(PowGradFn.f_d, x, y)

    @staticmethod
    def f_d(x: Tensor, y: Tensor, grad: ndarray, *args, **kwargs) -> (ndarray, ndarray):
        b = x.arr**y.arr * grad
        d_x = b * y.arr / x.arr
        d_y = b * np.log(x.arr)
        return d_x, d_y

class RPowGradFn(GradFn):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(RPowGradFn.f_d, x, y)

    @staticmethod
    def f_d(x: Tensor, y: Tensor, grad: ndarray, *args, **kwargs) -> (ndarray, ndarray):
        b = y.arr**x.arr * grad
        d_x = b * np.log(y.arr)
        d_y = b * x.arr / y.arr
        return d_x, d_y

class MatmulGradFn(GradFn):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(MatmulGradFn.f_d, x, y)

    @staticmethod
    def f_d(x: Tensor, y: Tensor, grad: ndarray, *args, **kwargs) -> (ndarray, ndarray):
        d_x = grad @ np.moveaxis(y.arr, -1, -2)
        d_y = np.moveaxis(x.arr, -1, -2) @ grad
        return d_x, d_y

class RMatmulGradFn(GradFn):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(RMatmulGradFn.f_d, x, y)

    @staticmethod
    def f_d(x: Tensor, y: Tensor, grad: ndarray, *args, **kwargs) -> (ndarray, ndarray):
        d_x = np.moveaxis(y.arr, -1, -2) @ grad
        d_y = grad @ np.moveaxis(x.arr, -1, -2)
        return d_x, d_y