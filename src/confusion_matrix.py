from typing import Literal

import torch
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat, to_onehot


class ConfusionMatrix(Metric):
    def __init__(
        self,
        metric: Literal["precision", "recall"],
        threshold_iou: float,
        num_classes: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.metric = metric
        self.threshold_iou = threshold_iou
        self.num_classes = num_classes
        self.add_state("tp", default=[], dist_reduce_fx="cat")
        self.add_state("fp", default=[], dist_reduce_fx="cat")
        self.add_state("fn", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # both tensors have shape BHW
        preds_one_hot = to_onehot(preds, self.num_classes)
        target_one_hot = to_onehot(target, self.num_classes)

        intersection = torch.sum(preds_one_hot * target_one_hot, dim=(2, 3))
        union = torch.sum(preds_one_hot + target_one_hot, dim=(2, 3)) - intersection

        iou = intersection / union
        is_found = iou > self.threshold_iou

        any_target = target_one_hot.sum(dim=(2, 3)) > 0
        any_preds = preds_one_hot.sum(dim=(2, 3)) > 0

        self.tp.append(is_found.sum(dim=0, keepdim=True))
        self.fp.append((~is_found * any_target).sum(dim=0, keepdim=True))
        self.fn.append((~is_found * any_preds).sum(dim=0, keepdim=True))

    def compute(self) -> torch.Tensor:
        tp = dim_zero_cat(self.tp)
        fp = dim_zero_cat(self.fp)
        fn = dim_zero_cat(self.fn)

        if self.metric == "precision":
            return torch.nansum(tp / (tp + fp))
        elif self.metric == "recall":
            return torch.nansum(tp / (tp + fn))
        else:
            raise NotImplementedError("Unknown metric option.")
