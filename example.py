import warnings
from typing import List
import numpy as np
from numpy import ndarray
from numpytorch import Tensor
from numpytorch.functions import *
from numpytorch.optim import SGDOptimizer


warnings.filterwarnings("ignore")


class Linear:
    def __init__(self, d_in: int, d_out: int) -> None:
        self.weight = rand(d_in, d_out, requires_grad=True)
        self.bias = rand(d_out, requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

    def parameters(self) -> List[Tensor]:
        return [self.weight, self.bias]

class Model:
    def __init__(self) -> None:
        self.layer0 = Linear(4, 20)
        self.layer1 = Linear(20, 20)
        self.layer2 = Linear(20, 1)

    def __call__(self, x: Tensor) -> Tensor:
        out = self.layer0(x)
        out = relu(out)
        out = self.layer1(out)
        out = relu(out)
        out = self.layer2(out)
        return out

    def parameters(self) -> List[Tensor]:
        return [
            *self.layer0.parameters(),
            *self.layer1.parameters(),
            *self.layer2.parameters()
        ]


def f_label(x: ndarray) -> ndarray:
    w = np.random.rand(4)
    e = np.array([1, 2, 1, 2])
    return (w * x ** e).sum(-1, keepdims=True)

train_x = np.mgrid[
    -1:1:0.1,
    -1:1:0.1,
    -1:1:0.1,
    -1:1:0.1,
].reshape(-1, 4)
train_y = f_label(train_x)


model = Model()
optimizer = SGDOptimizer(model.parameters(), lr=1e-04)
def criterion(y_pred: Tensor, y: Tensor) -> Tensor:
    return mean((y_pred - y) ** 2)


batch_size = 500
n_print = 20

buf = 0
for i in range(1, 2000):
    optimizer.zero_grad()

    idx = np.random.permutation(train_y.shape[0])[:batch_size]
    x = Tensor(train_x[idx])
    y = Tensor(train_y[idx])
    y_pred = model(x)

    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

    buf += loss.arr.item()
    if i % n_print == 0:
        print(buf / n_print)
        buf = 0