from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Optional, Union

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.trainer.states import TrainerFn
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from kwja.datamodule.datasets import (
    CharDataset,
    CharInferenceDataset,
    Seq2SeqDataset,
    Seq2SeqInferenceDataset,
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
        self.valid_datasets: Dict[str, Union[TypoDataset, Seq2SeqDataset, CharDataset, WordDataset]] = {}
        self.test_datasets: Dict[str, Union[TypoDataset, Seq2SeqDataset, CharDataset, WordDataset]] = {}
        self.predict_dataset: Optional[
            Union[
                TypoDataset,
                Seq2SeqDataset,
                CharDataset,
                WordDataset,
                TypoInferenceDataset,
                Seq2SeqInferenceDataset,
                CharInferenceDataset,
                WordInferenceDataset,
            ]
        ] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == TrainerFn.FITTING:
            self.train_dataset = ConcatDataset(hydra.utils.instantiate(config) for config in self.cfg.train.values())
        if stage in (TrainerFn.FITTING, TrainerFn.VALIDATING, TrainerFn.TESTING):
            self.valid_datasets = {corpus: hydra.utils.instantiate(config) for corpus, config in self.cfg.valid.items()}
        if stage == TrainerFn.TESTING:
            self.test_datasets = {corpus: hydra.utils.instantiate(config) for corpus, config in self.cfg.test.items()}
        if stage == TrainerFn.PREDICTING:
            self.predict_dataset = hydra.utils.instantiate(self.cfg.predict)

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return self._get_dataloader(dataset=self.train_dataset, shuffle=True)

    def val_dataloader(self) -> List[DataLoader]:
        return [self._get_dataloader(dataset, shuffle=False) for dataset in self.valid_datasets.values()]

    def test_dataloader(self) -> List[DataLoader]:
        return [self._get_dataloader(dataset, shuffle=False) for dataset in self.test_datasets.values()]

    def predict_dataloader(self) -> DataLoader:
        assert isinstance(self.predict_dataset, Dataset)
        return self._get_dataloader(self.predict_dataset, shuffle=False)

    def _get_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=dataclass_data_collator,
            pin_memory=True,
        )


def dataclass_data_collator(features: List[Any]) -> Dict[str, Union[Tensor, List[str]]]:
    first: Any = features[0]
    assert is_dataclass(first), "Data must be a dataclass"
    batch: Dict[str, Union[Tensor, List[str]]] = {}
    for field in fields(first):
        feats = [getattr(f, field.name) for f in features]
        if "text" in field.name:
            batch[field.name] = feats
        else:
            batch[field.name] = torch.as_tensor(feats)
    return batch
