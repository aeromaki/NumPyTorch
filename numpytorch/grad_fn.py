import numpy as np
from numpy import ndarray
from typing import Any, Callable, Tuple


Tensor = "Tensor"

class GradFn:
    def __init__(
        self,
        x: Tensor,
        y: Tensor,
        f_d: Callable[[Tensor, Tensor, ndarray], Tuple[ndarray, ndarray]]
    ) -> None:
        self.x = x
        self.y = y
        self.f_d = f_d

    def __call__(self, *args) -> None:
        self.propagate(*args)

    def propagate(self, grad: ndarray) -> None:
        d_x, d_y = self.f_d(self.x, self.y, grad)
        if d_x.shape != self.x.arr.shape:
            d_x = d_x.sum(0)
        if d_y.shape != self.y.arr.shape:
            d_y = d_y.sum(0)
        if self.x.requires_grad:
            if self.x.grad is not None:
                self.x.grad += d_x
            else:
                self.x.grad = d_x
            if self.x.grad_fn is not None:
                self.x.grad_fn(self.x.grad)
        if self.y.requires_grad:
            if self.y.grad is not None:
                self.y.grad += d_y
            else:
                self.y.grad = d_y
            if self.y.grad_fn is not None:
                self.y.grad_fn(self.y.grad)

class AddGradFn(GradFn):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y, AddGradFn.f_d)

    @staticmethod
    def f_d(x: Tensor, y: Tensor, grad: ndarray) -> (ndarray, ndarray):
        d_x = np.ones_like(x.arr) * grad
        d_y = np.ones_like(y.arr) * grad
        return d_x, d_y

class SubGradFn(GradFn):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y, SubGradFn.f_d)

    @staticmethod
    def f_d(x: Tensor, y: Tensor, grad: ndarray) -> (ndarray, ndarray):
        d_x = np.ones_like(x.arr) * grad
        d_y = -np.ones_like(y.arr) * grad
        return d_x, d_y

class RSubGradFn(GradFn):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y, RSubGradFn.f_d)

    @staticmethod
    def f_d(x: Tensor, y: Tensor, grad: ndarray) -> (ndarray, ndarray):
        d_x = -np.ones_like(x.arr) * grad
        d_y = np.ones_like(y.arr) * grad
        return d_x, d_y

class MulGradFn(GradFn):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y, MulGradFn.f_d)

    @staticmethod
    def f_d(x: Tensor, y: Tensor, grad: ndarray) -> (ndarray, ndarray):
        d_x = y.arr * grad
        d_y = x.arr * grad
        return d_x, d_y

class DivGradFn(GradFn):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y, DivGradFn.f_d)

    @staticmethod
    def f_d(x: Tensor, y: Tensor, grad: ndarray) -> (ndarray, ndarray):
        d_x = grad / y.arr
        d_y = -x.arr / y.arr**2 * grad
        return d_x, d_y

class RDivGradFn(GradFn):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y, RDivGradFn.f_d)

    @staticmethod
    def f_d(x: Tensor, y: Tensor, grad: ndarray) -> (ndarray, ndarray):
        d_x = -y.arr / x.arr**2 * grad
        d_y = grad / x.arr
        return d_x, d_y

class PowGradFn(GradFn):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y, PowGradFn.f_d)

    @staticmethod
    def f_d(x: Tensor, y: Tensor, grad: ndarray) -> (ndarray, ndarray):
        b = x.arr**y.arr * grad
        d_x = b * y.arr / x.arr
        d_y = b * np.log(x.arr)
        return d_x, d_y

class RPowGradFn(GradFn):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y, RPowGradFn.f_d)

    @staticmethod
    def f_d(x: Tensor, y: Tensor, grad: ndarray) -> (ndarray, ndarray):
        b = y.arr**x.arr * grad
        d_x = b * np.log(y.arr)
        d_y = b * x.arr / y.arr
        return d_x, d_y

class MatmulGradFn(GradFn):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y, MatmulGradFn.f_d)

    @staticmethod
    def f_d(x: Tensor, y: Tensor, grad: ndarray) -> (ndarray, ndarray):
        d_x = grad @ np.moveaxis(y.arr, -1, -2)
        d_y = np.moveaxis(x.arr, -1, -2) @ grad
        return d_x, d_y

class RMatmulGradFn(GradFn):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y, RMatmulGradFn.f_d)

    @staticmethod
    def f_d(x: Tensor, y: Tensor, grad: ndarray) -> (ndarray, ndarray):
        d_x = np.moveaxis(y.arr, -1, -2) @ grad
        d_y = grad @ np.moveaxis(x.arr, -1, -2)
        return d_x, d_y