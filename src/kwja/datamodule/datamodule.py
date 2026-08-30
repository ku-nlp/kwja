import json
from dataclasses import fields, is_dataclass
from typing import Any, TypeAlias

import hydra
import lightning as L
import torch
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import DictConfig, OmegaConf
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
from kwja.utils.constants import IGNORE_INDEX

# Process-wide cache of instantiated tokenizers, keyed by their resolved config.
_TOKENIZER_CACHE: dict[str, Any] = {}

DatasetType: TypeAlias = (
    TypoDataset
    | Seq2SeqDataset
    | CharDataset
    | WordDataset
    | TypoInferenceDataset
    | Seq2SeqInferenceDataset
    | CharInferenceDataset
    | WordInferenceDataset
)


def _instantiate_tokenizer_cached(tokenizer_cfg: DictConfig) -> Any:
    resolved_config = OmegaConf.to_container(
        tokenizer_cfg,
        resolve=True,
        enum_to_str=True,
    )
    key = json.dumps(resolved_config, sort_keys=True, ensure_ascii=False)
    if key not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[key] = hydra.utils.instantiate(tokenizer_cfg)
    return _TOKENIZER_CACHE[key]


class DataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg

        self.train_dataset: ConcatDataset | None = None
        self.valid_datasets: dict[str, DatasetType] = {}
        self.test_datasets: dict[str, DatasetType] = {}
        self.predict_dataset: DatasetType | None = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str | None = None) -> None:
        if stage == TrainerFn.FITTING:
            self.train_dataset = ConcatDataset(self._get_dataset(config) for config in self.cfg.train.values())
        if stage in (TrainerFn.FITTING, TrainerFn.VALIDATING, TrainerFn.TESTING):
            self.valid_datasets = {corpus: self._get_dataset(config) for corpus, config in self.cfg.valid.items()}
        if stage == TrainerFn.TESTING:
            self.test_datasets = {corpus: self._get_dataset(config) for corpus, config in self.cfg.test.items()}
        if stage == TrainerFn.PREDICTING:
            self.predict_dataset = self._get_dataset(self.cfg.predict)

    @staticmethod
    def _get_dataset(config: DictConfig) -> DatasetType:
        tokenizer = _instantiate_tokenizer_cached(config.tokenizer)
        return hydra.utils.instantiate(config, tokenizer=tokenizer)

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> dict[str, DataLoader]:
        return {corpus: self._get_dataloader(dataset, shuffle=False) for corpus, dataset in self.valid_datasets.items()}

    def test_dataloader(self) -> dict[str, DataLoader]:
        return {corpus: self._get_dataloader(dataset, shuffle=False) for corpus, dataset in self.test_datasets.items()}

    def predict_dataloader(self) -> DataLoader:
        assert isinstance(self.predict_dataset, Dataset)
        return self._get_dataloader(self.predict_dataset, shuffle=False)

    def _get_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        if self.cfg.dataset_type == "word":
            collate_fn = word_dataclass_data_collator
        else:
            collate_fn = token_dataclass_data_collator
        return DataLoader(
            dataset=dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )


def token_dataclass_data_collator(batch_features: list[Any]) -> dict[str, Tensor | list[str]]:
    first_features: Any = batch_features[0]
    assert is_dataclass(first_features), "Data must be a dataclass"

    token_indices = torch.arange(max(sum(fs.attention_mask) for fs in batch_features))

    batch: dict[str, Tensor | list[str]] = {}
    for field in fields(first_features):
        features = [getattr(fs, field.name) for fs in batch_features]
        if field.name in {"surfs"}:
            batch[field.name] = features
        else:
            value = torch.as_tensor(features)
            if value.ndim == 1 or value.size(1) == 0:
                pass
            elif field.name in {"seq2seq_labels"}:
                target_indices = torch.arange(value.ne(IGNORE_INDEX).sum(dim=1).max().item())
                value = value[:, target_indices]
            else:
                value = value[:, token_indices]
            batch[field.name] = value
    return batch


def word_dataclass_data_collator(batch_features: list[Any]) -> dict[str, Tensor | list[str]]:
    first_features: Any = batch_features[0]
    assert is_dataclass(first_features), "Data must be a dataclass"

    token_indices = torch.arange(max(sum(fs.attention_mask) for fs in batch_features))
    word_indices = torch.arange(max(sum(any(row) for row in fs.subword_map) for fs in batch_features))

    batch: dict[str, Tensor | list[str]] = {}
    for field in fields(first_features):
        features = [getattr(fs, field.name) for fs in batch_features]
        value = torch.as_tensor(features)
        if value.ndim == 1 or value.size(1) == 0:
            pass
        elif field.name in {"input_ids", "attention_mask", "reading_labels"}:
            value = value[:, token_indices]
        elif field.name in {"subword_map", "reading_subword_map"}:
            value = value[:, word_indices.unsqueeze(1), token_indices]
        elif field.name in {"dependency_mask", "cohesion_mask", "cohesion_labels", "discourse_labels"}:
            value = value[..., word_indices.unsqueeze(1), word_indices]
        elif field.name == "special_token_indices":
            pass
        else:
            value = value[:, word_indices]
        batch[field.name] = value
    return batch
