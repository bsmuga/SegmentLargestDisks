from argparse import ArgumentParser

import lightning as L
from torch.utils.data import DataLoader

from deep_circle_counter.dataset import CircleDataset
from deep_circle_counter.segmentation_model import SegmentationModel


def main(hparams: dict[str, str]) -> None:
    trainer = L.Trainer()

    model = SegmentationModel(**hparams["model"])

    loader_train = DataLoader(CircleDataset(**hparams["dataset_train"]))
    loader_valid = DataLoader(CircleDataset(**hparams["dataset_valid"]))
    loader_test = DataLoader(CircleDataset(**hparams["dataset_test"]))

    trainer.fit(
        model=model, train_dataloaders=loader_train, val_dataloaders=loader_valid
    )
    trainer.test(model=model, dataloaders=loader_test)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset_train", default=None)
    parser.add_argument("--dataset_valid", default=None)
    parser.add_argument("--dataset_test", default=None)
    args = parser.parse_args()

    main(args)
