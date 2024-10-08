import torch
from torchmetrics import MetricCollection
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU

from deep_circle_counter.metrics.accuracy import MyAccuracy


class Metric:
    def __init__(self, num_classes: int) -> None:
        self.metric = MetricCollection(
            [
                MyAccuracy(num_classes),
                GeneralizedDiceScore(num_classes),
                MeanIoU(num_classes),
            ]
        )

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        self.metric.update(prediction, target)

    @property
    def summary(self) -> dict[str, torch.Tensor]:
        return self.metric.compute()
