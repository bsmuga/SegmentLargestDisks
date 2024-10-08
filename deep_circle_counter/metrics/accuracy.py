import torch
from torchmetrics import Metric


class MyAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        raise NotImplementedError

    def compute(self) -> torch.Tensor:
        raise NotImplementedError
