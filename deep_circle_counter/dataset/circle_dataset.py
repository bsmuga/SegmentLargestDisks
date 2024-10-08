import pandas as pd
import torch
from torch.utils.data import Dataset


class CircleDataset(Dataset):
    def __init__(self, table_path: str) -> None:
        self.table = pd.read_csv(table_path)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    @classmethod
    def generate_table(cls, num_samples: int, table_path: str) -> "CircleDataset":
        raise NotImplementedError
