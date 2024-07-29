from numpy import ndarray

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Callable, Optional
)
if TYPE_CHECKING:
    from numpytorch.tensor import TensorNode


class GradFn(ABC):
    def __init__(self, *args: 'TensorNode') -> None:
        self.nodes: tuple['TensorNode', ...] = args

    def __call__(self, y: 'TensorNode') -> None:
        self.propagate(y)

    @abstractmethod
    def f_d(self, *args: 'TensorNode') -> tuple[ndarray, ...]:
        ...

    @staticmethod
    def _handle_broadcast(x: 'TensorNode', dx: ndarray) -> ndarray:
        if dx.ndim > x.ndim:
            assert dx.shape[-x.ndim:] == x.shape or x.shape == ()
            dx = dx.reshape(-1, *x.shape).sum(0)
        else:
            assert dx.ndim == x.ndim
            for i, (n_dx, n_x) in enumerate(zip(dx.shape, x.shape)):
                if n_x == 1:
                    dx = dx.sum(i, keepdims=True)
        return dx

    def propagate(self, y: 'TensorNode') -> None:
        grads: tuple[ndarray, ...] = self.f_d(*self.nodes, y)
        for x, dx in zip(self.nodes, grads):
            if x.requires_grad:
                if x.shape != dx.shape:
                    dx = self._handle_broadcast(x, dx)

                if x.grad is not None:
                    x.grad += dx
                else:
                    x.grad = dx

                x.grad_cnt -= 1
                if x.grad_fn is not None and x.grad_cnt == 0:
                    x.grad_fn(x)