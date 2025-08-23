import numpy as np


def to_onehot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels to one-hot encoding.
    
    Parameters
    ----------
    labels : np.ndarray
        Integer labels array of shape (B, H, W)
    num_classes : int
        Number of classes
        
    Returns
    -------
    np.ndarray
        One-hot encoded array of shape (B, num_classes, H, W)
    """
    B, H, W = labels.shape
    one_hot = np.zeros((B, num_classes, H, W), dtype=np.float32)
    
    for b in range(B):
        for c in range(num_classes):
            one_hot[b, c] = (labels[b] == c).astype(np.float32)
    
    return one_hot


class ConfusionMatrix:
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
    ):
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, 4), dtype=np.float32)

    def update(self, preds: np.ndarray, target: np.ndarray) -> None:
        """Update confusion matrix with new predictions and targets.
        
        Parameters
        ----------
        preds : np.ndarray
            Predictions array of shape (B, H, W)
        target : np.ndarray
            Target array of shape (B, H, W)
        """
        # Ensure inputs are 3D (batch, height, width)
        if preds.ndim == 2:
            preds = preds[np.newaxis, ...]
        if target.ndim == 2:
            target = target[np.newaxis, ...]
            
        # Convert to one-hot encoding
        preds_one_hot = to_onehot(preds, self.num_classes)
        target_one_hot = to_onehot(target, self.num_classes)

        # Calculate presence, intersection, and union
        target_present = np.sum(target_one_hot, axis=(2, 3)) > 0
        preds_present = np.sum(preds_one_hot, axis=(2, 3)) > 0
        intersection = np.sum(preds_one_hot * target_one_hot, axis=(2, 3))
        union = np.sum(preds_one_hot + target_one_hot, axis=(2, 3)) - intersection

        # Calculate IoU and determine if found
        # Handle division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = np.where(union > 0, intersection / union, 0)
        is_found = iou > self.iou_threshold

        # Update confusion matrix
        self.confusion_matrix[:, 0] += is_found.sum(axis=0)
        self.confusion_matrix[:, 1] += np.logical_and(
            intersection == 0, preds_present
        ).sum(axis=0)
        self.confusion_matrix[:, 2] += np.logical_and(~is_found, target_present).sum(
            axis=0
        )
        self.confusion_matrix[:, 3] += np.logical_and(
            ~target_present, ~preds_present
        ).sum(axis=0)

    def compute(self) -> np.ndarray:
        """Return the confusion matrix.
        
        Returns
        -------
        np.ndarray
            Confusion matrix of shape (num_classes, 4)
        """
        return self.confusion_matrix
    
    def reset(self) -> None:
        """Reset the confusion matrix to zeros."""
        self.confusion_matrix = np.zeros((self.num_classes, 4), dtype=np.float32)
        
    def __call__(self, preds: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Compute confusion matrix for a single batch.
        
        Parameters
        ----------
        preds : np.ndarray
            Predictions array
        target : np.ndarray
            Target array
            
        Returns
        -------
        np.ndarray
            Confusion matrix
        """
        # Create a temporary instance to compute for this batch only
        temp = ConfusionMatrix(self.iou_threshold, self.num_classes)
        temp.update(preds, target)
        return temp.compute()