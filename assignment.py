from numpytorch import Tensor, nn
from numpytorch.functions import *


class MNISTClassificationModel(nn.Module):
    def __init__(self) -> None:
        self.seq = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = reshape(x, (x.shape[0], -1))
        logits = self.seq(x)
        return logits