from typing import Optional, Union

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.trainer.states import TrainerFn
from torch.utils.data import DataLoader

from jula.datamodule.datasets.char_dataset import CharDataset
from jula.datamodule.datasets.custom_concat_dataset import CustomConcatDataset
from jula.datamodule.datasets.typo_dataset import TypoDataset
from jula.datamodule.datasets.word_dataset import WordDataset


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg
        self.batch_size: int = cfg.batch_size
        self.num_workers: int = cfg.num_workers

        self.train_dataset: CustomConcatDataset = None
        self.valid_datasets: dict[
            str, Union[CharDataset, TypoDataset, WordDataset]
        ] = {}
        self.test_datasets: dict[str, Union[CharDataset, TypoDataset, WordDataset]] = {}

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (TrainerFn.FITTING, TrainerFn.TUNING):
            self.train_dataset = CustomConcatDataset(
                [
                    hydra.utils.instantiate(config)
                    for config in self.cfg.dataset.train.values()
                ]
            )
        if stage in (
            TrainerFn.FITTING,
            TrainerFn.TUNING,
            TrainerFn.VALIDATING,
            TrainerFn.TESTING,
            TrainerFn.PREDICTING,
        ):
            self.valid_datasets = {
                corpus: hydra.utils.instantiate(config)
                for corpus, config in self.cfg.dataset.valid.items()
            }
        if stage in (TrainerFn.TESTING, TrainerFn.PREDICTING):
            self.test_datasets = {
                corpus: hydra.utils.instantiate(config)
                for corpus, config in self.cfg.dataset.test.items()
            }

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(dataset=self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return [
            self._get_dataloader(dataset, shuffle=False)
            for dataset in self.valid_datasets.values()
        ]

    def test_dataloader(self) -> DataLoader:
        return [
            self._get_dataloader(dataset, shuffle=False)
            for dataset in self.test_datasets.values()
        ]

    def predict_dataloader(self) -> DataLoader:
        return [
            self._get_dataloader(dataset, shuffle=False)
            for dataset in self.test_datasets.values()
        ]

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
