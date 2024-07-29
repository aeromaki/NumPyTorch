import numpy as np
from numpy import ndarray

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from numpytorch.tensor import Tensor
from .base import GradFn
from ._operators import _clip_eps


class SumGradFn(GradFn):
    def __init__(
        self,
        x: 'Tensor',
        axis: Optional[int | tuple[int, ...]] = None,
        keepdims: bool = False
    ) -> None:
        super().__init__(x)
        self.axis = axis
        self.keepdims = keepdims

    def f_d(self, *args: 'Tensor') -> tuple[ndarray]:
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
    def f_d(*args: 'Tensor') -> tuple[ndarray]:
        x, y = args
        assert y.grad is not None

        dx = (x.arr > 0) * y.grad
        return (dx,)

class LogGradFn(GradFn):
    def __init__(self, x: 'Tensor') -> None:
        super().__init__(x)

    @staticmethod
    def f_d(*args: 'Tensor') -> tuple[ndarray]:
        x, y = args
        assert y.grad is not None

        dx = y.grad / _clip_eps(x.arr)
        return (dx,)

class SigmoidGradFn(GradFn):
    def __init__(self, x: 'Tensor') -> None:
        super().__init__(x)

    @staticmethod
    def f_d(*args: 'Tensor') -> tuple[ndarray]:
        x, y = args
        assert y.grad is not None

        dx = y.arr * (1 - y.arr) * y.grad
        return (dx,)

class TanhGradFn(GradFn):
    def __init__(self, x: 'Tensor') -> None:
        super().__init__(x)

    @staticmethod
    def f_d(*args: 'Tensor') -> tuple[ndarray]:
        x, y = args
        assert y.grad is not None

        dx = (1 - y.arr)**2 * y.grad
        return (dx,)

class ReshapeGradFn(GradFn):
    def __init__(self, x: 'Tensor') -> None:
        super().__init__(x)

    @staticmethod
    def f_d(*args: 'Tensor') -> tuple[ndarray]:
        x, y = args
        assert y.grad is not None

        dx = y.grad.reshape(x.shape)
        return (dx,)

class RepeatGradFn(GradFn):
    def __init__(self, x: 'Tensor') -> None:
        super().__init__(x)

    @staticmethod
    def f_d(*args: 'Tensor') -> tuple[ndarray]:
        x, y = args
        assert y.grad is not None

        dx = y.grad.reshape(*x.shape, -1).sum(-1)

        return (dx,)