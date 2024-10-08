import argparse
import json

import torch

from deep_circle_counter.dataset import CircleDataset
from deep_circle_counter.loops import test_loop, train_loop
from deep_circle_counter.losses import Loss
from deep_circle_counter.metrics import Metric
from deep_circle_counter.models import FPN, SegFormer, Unet


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Training")
    parser.add_argument(
        "--model", type=int, choices=["unet", "fpn", "segformer"], required=True
    )
    parser.add_argument("-mp", "--model-params", type=str, required=True)
    parser.add_argument("-dp", "--dataset-params", type=str, required=True)
    parser.add_argument("-nc", "--num_classes", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("-lr", "--learning-rate", type=float, required=True)
    parser.add_argument("--losses-weights", nargs=3, type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = {"unet": Unet, "fpn": FPN, "segformer": SegFormer}

    try:
        model_params = json.loads(args.model_params)
        dataset_params = json.loads(args.dataset_params)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON format: {e}")
        exit(1)

    model = models[args.model](**model_params)
    dataset = CircleDataset(**dataset_params)

    loss_fn = Loss(**args.losses_weights)
    metric_fn = Metric(args.num_classes)

    if torch.cuda.is_available():
        model = model.to("cuda")

    train_loop(model, dataset, loss_fn, args.num_epochs, args.learning_rate)
    test_loop(model, dataset, metric_fn)


if __name__ == "__main__":
    main()
