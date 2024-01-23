import numpytorch as npt
from numpytorch import Tensor, rand, nn
from numpytorch.optim import SGDOptimizer

class CustomModel(nn.Module):
    def __init__(self):
        self.seq = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)

model = CustomModel()

optimizer = SGDOptimizer(model.parameters(), lr=1e-04)

for i in range(100):
    optimizer.zero_grad()
    x = rand(20, 30, 5)
    y = model(x)
    b = npt.mean(y)
    b.backward()
    optimizer.step()
    print(b.arr.item())