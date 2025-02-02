import torch
from torchmetrics import Metric
from torchmetrics.utilities.data import to_onehot


class ConfusionMatrix(Metric):
    """Confusion Matrix that consider IoU
    to determine true positives, false positives,
    false negatives and true negatives.

    If prediction intersect ground truth and IoU
    exceed threshold it is considered as true positive,
    If ground truth is present, but IoU is not exceeded,
    it is considered as false negative,
    If predicition is present and IoU is not exceeded
    and intersection with ground truth is equal to zero,
    it is considered as false positive,
    and if prediction and ground truth are not present,
    it considered as true negative.

    Parameters
    ----------
    iou_threshold : float
        threshold of iou (intersection over union)
        to determine if class is considered as detected
    num_classes : int
        Number of labels including background.

    Note: confusion matrix is represented as (num_classes, 4) array
    where per class are encoded tp, fp, fn, tn.
    """

    def __init__(
        self,
        iou_threshold: float,
        num_classes: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        self.add_state(
            "confusion_matrix",
            default=torch.zeros((num_classes, 4)),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # both tensors have shape BHW
        preds_one_hot = to_onehot(preds, self.num_classes)
        target_one_hot = to_onehot(target, self.num_classes)

        target_present = torch.sum(target_one_hot, dim=(2, 3)) > 0
        preds_present = torch.sum(preds_one_hot, dim=(2, 3)) > 0
        intersection = torch.sum(preds_one_hot * target_one_hot, dim=(2, 3))
        union = torch.sum(preds_one_hot + target_one_hot, dim=(2, 3)) - intersection

        iou = intersection / union
        is_found = iou > self.iou_threshold

        self.confusion_matrix[:, 0] += is_found.sum(dim=0)
        self.confusion_matrix[:, 1] += torch.logical_and(
            intersection == 0, preds_present
        ).sum(dim=0)
        self.confusion_matrix[:, 2] += torch.logical_and(~is_found, target_present).sum(
            dim=0
        )
        self.confusion_matrix[:, 3] += torch.logical_and(
            ~target_present, ~preds_present
        ).sum(dim=0)

    def compute(self) -> torch.Tensor:
        return self.confusion_matrix
