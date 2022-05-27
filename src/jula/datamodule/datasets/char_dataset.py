import torch

from jula.datamodule.datasets.base_dataset import BaseDataset


class CharDataset(BaseDataset):
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError
