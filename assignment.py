import numpy as np
from numpy import ndarray
from typing import Tuple, Union

import numpytorch as npt
from numpytorch import Tensor, nn
from numpytorch.autograd import GradFn


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

"""
# Your model
class MNISTClassificationModel(nn.Module):
    def __init__(self) -> None:
        pass

    def forward(self, x: Tensor) -> Tensor:
        # Input shape: (batch_size, 1, 28, 28)
        # Return shape: (batch_size, 10)
        pass
"""


class Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.filter = nn.Parameter.new(out_channels, in_channels, kernel_size, kernel_size)
        self.bias = nn.Parameter.new(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        n, _, h, w = x.shape

        c = self.in_channels
        f = self.out_channels
        k = self.kernel_size

        h_ = h - k + 1
        w_ = w - k + 1

        out = npt.zeros(n, f, h_, w_)

        xr = npt.reshape(x, (n, 1, c, h, w))
        wr = npt.reshape(self.filter, (1, f, c, k, k))

        for i in range(h_):
            for j in range(h_):
                out[:, :, i, j] = npt.sum(xr[:, :, :, i:i+k, j:j+k] * wr, axis=(2, 3, 4))

        out += npt.reshape(self.bias, (1, f, 1, 1))

        return out


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int) -> None:
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: Tensor):
        n, c, h, w = x.shape

        h_ = 1 + (h - self.kernel_size) // self.stride
        w_ = 1 + (w - self.kernel_size) // self.stride

        out = npt.zeros(n, c, h_, w_)

        for i in range(h_):
            for j in range(w_):
                hs = i * self.stride
                ws = j * self.stride
                out[:, :, i, j] = npt.max(x[:, :, hs:hs+self.kernel_size, ws:ws+self.kernel_size], axis=(2, 3))

        return out


class MNISTClassificationModel(nn.Module):
    def __init__(self) -> None:
        self.seq = nn.Sequential(
            Conv2d(1, 8, 3),
            MaxPool2d(2, 2),
            Conv2d(8, 16, 3),
            MaxPool2d(2, 2),
            Conv2d(16, 32, 3)
        )
        self.linear = nn.Linear(288, 10)

    def forward(self, x: Tensor) -> Tensor:
        # Input shape: (batch_size, 1, 28, 28)
        # Return shape: (batch_size, 10)
        out = self.seq(x)
        out = npt.reshape(out, (x.shape[0], -1))
        out = self.linear(out)
        return out