from typing import Optional, Union

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.trainer.states import TrainerFn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from kwja.datamodule.datasets import (
    CharDataset,
    CharInferenceDataset,
    TypoDataset,
    TypoInferenceDataset,
    WordDataset,
    WordInferenceDataset,
)


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg
        self.batch_size: int = cfg.batch_size
        self.num_workers: int = cfg.num_workers

        self.train_dataset: Optional[ConcatDataset] = None
        self.valid_datasets: dict[str, Union[TypoDataset, CharDataset, WordDataset]] = {}
        self.test_datasets: dict[str, Union[TypoDataset, CharDataset, WordDataset]] = {}
        self.predict_dataset: Optional[
            Union[
                TypoDataset, CharDataset, WordDataset, TypoInferenceDataset, CharInferenceDataset, WordInferenceDataset
            ]
        ] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (TrainerFn.FITTING, TrainerFn.TUNING):
            self.train_dataset = ConcatDataset(hydra.utils.instantiate(config) for config in self.cfg.train.values())
        if stage in (TrainerFn.FITTING, TrainerFn.TUNING, TrainerFn.VALIDATING, TrainerFn.TESTING):
            self.valid_datasets = {corpus: hydra.utils.instantiate(config) for corpus, config in self.cfg.valid.items()}
        if stage in (TrainerFn.TESTING,):
            self.test_datasets = {corpus: hydra.utils.instantiate(config) for corpus, config in self.cfg.test.items()}
        if stage in (TrainerFn.PREDICTING,):
            self.predict_dataset = hydra.utils.instantiate(self.cfg.predict)

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(dataset=self.train_dataset, shuffle=True)

    def val_dataloader(self) -> list[DataLoader]:
        return [self._get_dataloader(dataset, shuffle=False) for dataset in self.valid_datasets.values()]

    def test_dataloader(self) -> list[DataLoader]:
        return [self._get_dataloader(dataset, shuffle=False) for dataset in self.test_datasets.values()]

    def predict_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.predict_dataset, shuffle=False)

    def _get_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )
