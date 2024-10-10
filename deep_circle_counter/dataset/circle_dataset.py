import pandas as pd
import torch
from torch.utils.data import Dataset

from deep_circle_counter.dataset.generate_circles import generate_circles

class CircleDataset(Dataset):
    def __init__(self, table_path: str) -> None:
        self.table = pd.read_csv(table_path)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    @staticmethod
    def generate_data(num_samples: int, table_path: str) -> pd.DataFrame:
        table = pd.DataFrame([generate_circles(200, 200, [10, 20, 25, 35])] for _ in range(num_samples))
        pass
