import pprint
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def train_loop(
    model: nn.Module,
    dataset: Dataset,
    loss: Callable,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    print_freq: int = 10,
) -> None:
    model.train()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(epochs)):
        batch_losses: list[torch.Tensor] = list()
        for input, target in dataloader:
            prediction = model(input)
            loss = loss(prediction, target)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_losses.append(loss.item())

        if epoch % print_freq == 0:
            tqdm.write(f"Epoch {epoch} loss: {torch.mean(batch_losses)/batch_size}")


@torch.no_grad
def test_loop(model: nn.Module, dataset: Dataset, metric_fn: Callable) -> None:
    model.eval()

    dataloader = DataLoader(dataset)
    for input, target in dataloader:
        metric_fn(input, target)

    pprint(f"Summary: {metric_fn.summary}")
