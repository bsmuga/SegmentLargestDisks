import lightning as L
from torch.utils.data import DataLoader

from src.data.dataset import DisksDataset


class DiskDataModule(L.LightningDataModule):
    def __init__(
        self,
        image_size: tuple[int, int],
        disk_max_num: int,
        labeled_disks: int,
        train_items: int,
        valid_items: int,
        test_items: int,
        batch_size: int,
        num_workers: int,
        seed: int = None,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.disk_max_num = disk_max_num
        self.labeled_disks = labeled_disks
        self.train_items = train_items
        self.valid_items = valid_items
        self.test_items = test_items
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage: str) -> None:
        # Use different seeds for train/val/test if seed is provided
        train_seed = self.seed if self.seed is None else self.seed
        valid_seed = self.seed if self.seed is None else self.seed + 1000
        test_seed = self.seed if self.seed is None else self.seed + 2000

        self.ds_train = DisksDataset(
            self.image_size,
            self.disk_max_num,
            self.labeled_disks,
            self.train_items,
            train_seed,
        )
        self.ds_valid = DisksDataset(
            self.image_size,
            self.disk_max_num,
            self.labeled_disks,
            self.valid_items,
            valid_seed,
        )
        self.ds_test = DisksDataset(
            self.image_size,
            self.disk_max_num,
            self.labeled_disks,
            self.test_items,
            test_seed,
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
