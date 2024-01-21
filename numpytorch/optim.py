from typing import Iterable
from tensor import Tensor

class SGDOptimizer:
    def __init__(self, params: Iterable[Tensor], lr: float) -> None:
        self.params = params
        self.lr = lr

    def step(self) -> None:
        for param in self.params:
            param.arr -= param.grad * self.lr

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None