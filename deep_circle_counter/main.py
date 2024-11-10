from deep_circle_counter.dataset import CircleDataset
from deep_circle_counter.trainer import Trainer


def main() -> None:
    # TODO: build simple CLI
    # check omegaconf
    args = ...

    dataset = CircleDataset(args.dataset)
    trainer = Trainer(args.trainer)

    trainer.train()
    trainer.test()


if __name__ == "__main__":
    main()
