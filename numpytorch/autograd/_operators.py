import numpy as np
from numpy import ndarray

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpytorch.tensor import TensorNode
from .grad_fn import GradFn


def _clip_eps(x: ndarray, eps: float = 1e-06) -> ndarray:
    return np.sign(x) * np.clip(np.abs(x), a_min=eps, a_max=None)


class AddGradFn(GradFn):
    def __init__(self, x0: 'TensorNode', x1: 'TensorNode') -> None:
        super().__init__(x0, x1)

    @staticmethod
    def f_d(*args: 'TensorNode') -> tuple[ndarray, ndarray]:
        x0, x1, y = args
        assert y.grad is not None

        dx0 = np.ones_like(x0.arr) * y.grad
        dx1 = np.ones_like(x1.arr) * y.grad
        return dx0, dx1

class SubGradFn(GradFn):
    def __init__(self, x0: 'TensorNode', x1: 'TensorNode') -> None:
        super().__init__(x0, x1)

    @staticmethod
    def f_d(*args: 'TensorNode') -> tuple[ndarray, ndarray]:
        x0, x1, y = args
        assert y.grad is not None

        dx0 = np.ones_like(x0.arr) * y.grad
        dx1 = -np.ones_like(x1.arr) * y.grad
        return dx0, dx1

class MulGradFn(GradFn):
    def __init__(self, x0: 'TensorNode', x1: 'TensorNode') -> None:
        super().__init__(x0, x1)

    @staticmethod
    def f_d(*args: 'TensorNode') -> tuple[ndarray, ndarray]:
        x0, x1, y = args
        assert y.grad is not None

        dx0 = x1.arr * y.grad
        dx1 = x0.arr * y.grad
        return dx0, dx1

class DivGradFn(GradFn):
    def __init__(self, x0: 'TensorNode', x1: 'TensorNode') -> None:
        super().__init__(x0, x1)

    @staticmethod
    def f_d(*args: 'TensorNode') -> tuple[ndarray, ndarray]:
        x0, x1, y = args
        assert y.grad is not None

        dx0 = y.grad / _clip_eps(x1.arr)
        dx1 = -x0.arr / _clip_eps(x1.arr**2) * y.grad
        return dx0, dx1

class PowGradFn(GradFn):
    def __init__(self, x0: 'TensorNode', x1: 'TensorNode') -> None:
        super().__init__(x0, x1)

    @staticmethod
    def f_d(*args: 'TensorNode') -> tuple[ndarray, ndarray]:
        x0, x1, y = args
        assert y.grad is not None
        # assert (x0.arr > 0).all()

        b = x0.arr**(x1.arr-1) * y.grad
        dx0 = x1.arr * b
        dx1 = np.log(x0.arr) * x0.arr * b
        return dx0, dx1

class MatmulGradFn(GradFn):
    def __init__(self, x0: 'TensorNode', x1: 'TensorNode') -> None:
        super().__init__(x0, x1)

    @staticmethod
    def f_d(*args: 'TensorNode') -> tuple[ndarray, ndarray]:
        x0, x1, y = args
        assert y.grad is not None

        dx0 = y.grad @ np.moveaxis(x1.arr, -1, -2)
        dx1 = np.moveaxis(x0.arr, -1, -2) @ y.grad
        return dx0, dx1

class RSubGradFn(SubGradFn):
    def __init__(self, x0: 'TensorNode', x1: 'TensorNode') -> None:
        super().__init__(x1, x0)

class RDivGradFn(DivGradFn):
    def __init__(self, x0: 'TensorNode', x1: 'TensorNode') -> None:
        super().__init__(x1, x0)

class RPowGradFn(PowGradFn):
    def __init__(self, x0: 'TensorNode', x1: 'TensorNode') -> None:
        super().__init__(x1, x0)

class RMatmulGradFn(MatmulGradFn):
    def __init__(self, x0: 'TensorNode', x1: 'TensorNode') -> None:
        super().__init__(x1, x0)

class GetitemGradFn(GradFn):
    def __init__(self, x: 'TensorNode', key) -> None:
        super().__init__(x)
        self.key = key

    def f_d(self, *args: 'TensorNode') -> tuple[ndarray]:
        x, y = args
        assert y.grad is not None

        dx = np.zeros_like(x.arr)
        dx[self.key] = y.grad
        return (dx,)

class SetitemGradFn(GradFn):
    def __init__(self, x: 'TensorNode', key) -> None:
        super().__init__(x)
        self.key = key

    def f_d(self, *args: 'TensorNode') -> tuple[ndarray]:
        x, y = args
        assert y.grad is not None

        dx = y.grad
        dx[self.key] = 0
        return (dx,)

class SetitemTensorGradFn(GradFn):
    def __init__(self, x: 'TensorNode', value: 'TensorNode', key) -> None:
        super().__init__(x, value)
        self.key = key

    def f_d(self, *args: 'TensorNode') -> tuple[ndarray, ...]:
        x, value, y = args
        assert y.grad is not None

        d_value: ndarray = y.grad[self.key].reshape(value.shape)

        dx = y.grad
        dx[self.key] = 0
        return (dx, d_value)