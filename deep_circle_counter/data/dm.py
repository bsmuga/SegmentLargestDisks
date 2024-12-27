import lightning as L
from torch.utils.data import DataLoader

from deep_circle_counter.data.dataset import CircleDataset


class CircleDataModule(L.LightningDataModule):
    def __init__(
        self,
        image_size: tuple[int, int],
        circles_max_num: int,
        labels: int,
        train_items: int,
        valid_items: int,
        test_items: int,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.circles_max_num = circles_max_num
        self.labels = labels
        self.train_items = train_items
        self.valid_items = valid_items
        self.test_items = test_items
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        self.ds_train = CircleDataset(
            self.image_size, self.circles_max_num, self.labels, self.train_items
        )
        self.ds_valid = CircleDataset(
            self.image_size, self.circles_max_num, self.labels, self.valid_items
        )
        self.ds_test = CircleDataset(
            self.image_size, self.circles_max_num, self.labels, self.test_items
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_valid, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
