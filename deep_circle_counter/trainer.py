import pprint
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from segmentation_models_pytorch import create_model, losses, metrics


class Trainer:
    def __init__(self):
        # TODO: use segmentation_models.pytorch models
        # model = create_model()
        pass

    def train(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: Callable,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        print_freq: int = 10,
    ) -> None:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in tqdm(range(epochs)):
            batch_losses: list[torch.Tensor] = list()
            for input, segmetation in dataloader:
                if torch.cuda.is_available():
                    input, segmetation = input.to("cuda"), segmetation.to("cuda")
                prediction = model(input)
                loss: torch.Tensor = loss_fn(prediction, segmetation)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_losses.append(loss)

            if epoch % print_freq == 0:
                tqdm.write(f"Epoch {epoch} loss: {torch.mean(batch_losses)/batch_size}")

    # @torch.no_grad
    def test(
        self, model: nn.Module, dataloader: DataLoader, metric_fn: Callable
    ) -> None:
        model.eval()

        for input, segmetation in dataloader:
            if torch.cuda.is_available():
                input, segmetation = input.to("cuda"), segmetation.to("cuda")
            prediction = model(input)
            metric_fn(prediction, segmetation)

        pprint(f"Summary: {metric_fn.summary}")
