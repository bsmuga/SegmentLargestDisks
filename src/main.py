import argparse

import lightning as L
import yaml
from lightning.pytorch.loggers import MLFlowLogger
from src.data import DiskDataModule
from src.segmentation_module import SegmentationModule


def main(hparams: dict[str, str | int | list]) -> None:
    model = SegmentationModule(**hparams["model"])
    data = DiskDataModule(**hparams["data"])
    logger = MLFlowLogger(**hparams["logger"])

    trainer = L.Trainer(**hparams["trainer"], logger=logger)
    trainer.fit(model=model, datamodule=data)
    trainer.test(datamodule=data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command-line interface for training a model."
    )
    parser.add_argument("--config", type=str, help="Path to configuration yaml.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    main(config)
