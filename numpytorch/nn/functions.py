from .base import Module
from numpytorch.functions import *


class ReLU(Module):
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return relu(x)

class Sigmoid(Module):
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return sigmoid(x)

class Tanh(Module):
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return tanh(x)

class CrossEntropyLoss(Module):
    @staticmethod
    def forward(logits: Tensor, q: Tensor) -> Tensor:
        if logits.shape != q.shape:
            q = one_hot(q, logits.shape[-1])
        log_p = logits - log(sum(exp(logits), -1, keepdims=True))
        ce = -sum(q * log_p, -1)
        return mean(ce)