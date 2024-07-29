from typing import Sequence

from numpytorch.tensor import Tensor


class SGD:
    def __init__(self, params: Sequence[Tensor], lr: float) -> None:
        self.params = params
        self.lr = lr

    def step(self) -> None:
        for param in self.params:
            if param.grad is not None:
                param.arr -= param.grad * self.lr

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None