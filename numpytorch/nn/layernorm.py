from .module import Module, Parameter
from numpytorch.tensor import Tensor
from numpytorch.functions import mean, var


class LayerNorm(Module):
    def __init__(
        self,
        eps: float = 1e-05
    ) -> None:
        self.eps = eps
        self.gamma = Parameter.new_scalar()
        self.beta = Parameter.new_scalar()

    def forward(self, x: Tensor) -> Tensor:
        return (x - mean(x, -1, keepdims=True)) / (var(x, -1) + self.eps) ** 0.5 * self.gamma + self.beta