import torch
from torch import nn

from deep_circle_counter.losses.dice import DiceLoss


class Loss(nn.Module):
    def __init__(self, dice_weight: float, ce_weight: float, mse_weight: float):
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

        norm = torch.norm([dice_weight, ce_weight, mse_weight])
        self.dice_weight = dice_weight / norm
        self.ce_weight = ce_weight / norm
        self.mse_weight = mse_weight / norm

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.dice_weight * self.mse(prediction, target) + self.ce_weight * self.ce
        )
