import numpy as np
from numpy import ndarray
from typing import Tuple, Union

from numpytorch import Tensor, nn
from numpytorch.grad_fn import GradFn
from numpytorch.functions import *


"""
Example model.
If you want to see how main.py works (before you finish the assignment),
try running it through this model.

class MNISTClassificationModel(nn.Module):
    def __init__(self) -> None:
        self.seq = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = reshape(x, (x.shape[0], -1))
        logits = self.seq(x)
        return logits
"""


# Convolution layer
class Conv2d(nn.Module):
    def __init__(self,) -> None:

        pass

        self.w: Tensor = None
        self.b: Tensor = None

    def forward(self, x: Tensor) -> Tensor:

        pass

        return Tensor(
            output,
            requires_grad=x.requires_grad,
            is_leaf=False,
            grad_fn=Conv2dGradFn(x, self.w, self.b)
        )


# Backward for convolution layer
class Conv2dGradFn(GradFn):
    def __init__(self, x: Tensor, w: Tensor, b: Tensor) -> None:
        super().__init__(x, w, b)

    def f_d(self, *args: Tensor) -> Tuple[ndarray, ndarray, ndarray]:
        x, w, b, y = args
        assert y.grad is not None

        pass

        return (dx, dw, db)


# Max pooling layer
class MaxPool2d(nn.Module):
    def __init__(self) -> None:

        pass

        self.h: int = None
        self.w: int = None
        self.stride: int = None

    def forward(self, x: Tensor) -> Tensor:

        pass

        return Tensor(
            output,
            requires_grad=x.requires_grad,
            is_leaf=False,
            grad_fn=MaxPool2dGradFn(x, h=self.h, w=self.w, stride=self.stride)
        )


# Backward for max pooling layer
class MaxPool2dGradFn(GradFn):
    def __init__(self, x: Tensor, h: int, w: int, stride: int) -> None:
        super().__init__(x)
        self.h = h
        self.w = w
        self.stride = stride

    def f_d(self, *args: Tensor) -> Tuple[ndarray]:
        x, y = args
        assert y.grad is not None

        pass

        return (dx,)


# Your model
class MNISTClassificationModel(nn.Module):
    def __init__(self) -> None:
        pass

    def forward(self, x: Tensor) -> Tensor:
        # Input shape: (batch_size, 1, 28, 28)
        # Return shape: (batch_size, 10)
        pass