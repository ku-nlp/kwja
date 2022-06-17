from typing import Optional, Union

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.trainer.states import TrainerFn
from torch.utils.data import DataLoader

from jula.datamodule.datasets.char_dataset import CharDataset
from jula.datamodule.datasets.typo_dataset import TypoDataset
from jula.datamodule.datasets.word_dataset import WordDataset


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg
        self.batch_size: int = cfg.batch_size
        self.num_workers: int = cfg.num_workers

        self.data_dir: str = cfg.dataset.data_dir

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (TrainerFn.FITTING, TrainerFn.TUNING):
            self.train_dataset = hydra.utils.instantiate(
                {"path": f"{self.data_dir}/train", **self.cfg.datamodule}
            )
        if stage in (
            TrainerFn.FITTING,
            TrainerFn.TUNING,
            TrainerFn.VALIDATING,
            TrainerFn.TESTING,
            TrainerFn.PREDICTING,
        ):
            self.valid_dataset = hydra.utils.instantiate(
                {"path": f"{self.data_dir}/valid", **self.cfg.datamodule}
            )
        if stage in (TrainerFn.TESTING, TrainerFn.PREDICTING):
            self.test_dataset = hydra.utils.instantiate(
                {"path": f"{self.data_dir}/test", **self.cfg.datamodule}
            )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(dataset=self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(dataset=self.valid_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(dataset=self.test_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._get_dataloader(dataset=self.test_dataset, shuffle=False)

    def _get_dataloader(
        self, dataset: Union[CharDataset, TypoDataset, WordDataset], shuffle: bool
    ) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )
