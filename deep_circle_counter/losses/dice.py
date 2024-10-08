import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self):
        raise NotImplementedError

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
