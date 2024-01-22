from typing import Iterable
from .tensor import Tensor

class SGDOptimizer:
    def __init__(self, params: Iterable[Tensor], lr: float) -> None:
        """
        Initializes a stochastic gradient descent (SGD) optimizer.

        How SGD works:
            - The gradient of the loss function with respect to the parameters is computed.
            - The parameters are updated by subtracting the gradient multiplied by the learning rate.

        Args:
            params (Iterable[Tensor]): The parameters to optimize.
            lr (float): The learning rate.

        Returns:
            None
        """
        self.params = params
        self.lr = lr

    def step(self) -> None:
        """
        Performs a single optimization step.

        Returns:
            None
        """
        for param in self.params:
            param.arr -= param.grad * self.lr

    def zero_grad(self) -> None:
        """
        Resets the gradients of all parameters to None.

        Returns:
            None
        """
        for param in self.params:
            param.grad = None