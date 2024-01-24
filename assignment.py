from numpytorch import Tensor, nn
from numpytorch.functions import *


def softmax(x: Tensor) -> Tensor:
    e = exp(x)
    return e / sum(e, -1, keepdims=True)

class MNISTClassificationModel(nn.Module):
    def __init__(self) -> None:
        n = 256
        self.seq = nn.Sequential(
            nn.Linear(784, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, 10)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = reshape(x, (x.shape[0], -1))
        logits = self.seq(x)
        return logits