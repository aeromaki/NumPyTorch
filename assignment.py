from numpytorch import Tensor, nn
from numpytorch.functions import *


class MNISTClassificationModel(nn.Module):
    def __init__(self) -> None:
        self.seq = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 10, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = reshape(x, (x.shape[0], -1))
        logits = self.seq(x)
        return logits