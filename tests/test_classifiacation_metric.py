import torch

from deep_circle_counter.classification_metric import ClassificationMetric


def test_classification_metric():
    target = torch.zeros((1, 100, 100))
    preds = torch.zeros((1, 100, 100))

    target[0, 10:20, 10:20] = 1
    target[0, 10:20, 70:80] = 2
    target[0, 70:80, 10:20] = 3

    preds[0, 10:20, 10:20] = 1
    preds[0, 10:20, 70:90] = 2
    preds[0, 70:80, 70:80] = 4

    precision = ClassificationMetric("precision", 0.5, 5)
    recall = ClassificationMetric("recall", 0.5, 5)

    precision.update(preds, target)
    precision.update(target, target)
    
    recall.update(preds, target)
    recall.update(target, target)

    precision_value = precision.compute()
    recall_value = recall.compute()
