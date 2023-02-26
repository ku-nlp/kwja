from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
from omegaconf import DictConfig
from transformers import PretrainedConfig, PreTrainedModel

from kwja.metrics import TypoModuleMetric
from kwja.modules.base import BaseModule
from kwja.modules.components.head import SequenceLabelingHead
from kwja.modules.functions.loss import compute_token_mean_loss
from kwja.utils.constants import RESOURCE_PATH


class TypoModule(BaseModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__(hparams)

        if valid_corpora := getattr(hparams.datamodule, "valid", None):
            self.valid_corpora: List[str] = list(valid_corpora)
            self.valid_corpus2typo_module_metric: Dict[str, TypoModuleMetric] = {
                corpus: TypoModuleMetric() for corpus in self.valid_corpora
            }
        if test_corpora := getattr(hparams.datamodule, "test", None):
            self.test_corpora: List[str] = list(test_corpora)
            self.test_corpus2typo_module_metric: Dict[str, TypoModuleMetric] = {
                corpus: TypoModuleMetric() for corpus in self.test_corpora
            }

        self.char_encoder: PreTrainedModel = hydra.utils.call(hparams.encoder)
        pretrained_model_config: PretrainedConfig = self.char_encoder.config
        if hasattr(hparams, "special_tokens"):
            self.char_encoder.resize_token_embeddings(pretrained_model_config.vocab_size + len(hparams.special_tokens))
        head_args: Tuple[int, float] = (
            pretrained_model_config.hidden_size,
            pretrained_model_config.hidden_dropout_prob,
        )

        self.kdr_tagger = SequenceLabelingHead(pretrained_model_config.vocab_size, *head_args)
        extended_vocab_size = sum(1 for _ in RESOURCE_PATH.joinpath("typo_correction", "multi_char_vocab.txt").open())
        self.ins_tagger = SequenceLabelingHead(pretrained_model_config.vocab_size + extended_vocab_size, *head_args)

    def forward(self, batch: Any) -> Dict[str, torch.Tensor]:
        truncation_length = self._truncate(batch)
        encoded = self.char_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        kdr_logits = self.kdr_tagger(encoded.last_hidden_state)
        ins_logits = self.ins_tagger(encoded.last_hidden_state)
        if truncation_length > 0:
            kdr_logits = self._pad(kdr_logits, truncation_length)
            ins_logits = self._pad(ins_logits, truncation_length)
        return {"kdr_logits": kdr_logits, "ins_logits": ins_logits}

    def _truncate(self, batch: Any) -> int:
        max_seq_length = batch["attention_mask"].sum(dim=1).max().item()
        for key, value in batch.items():
            if key in {"input_ids", "attention_mask"}:
                batch[key] = value[:, :max_seq_length].contiguous()
        return self.hparams.max_seq_length - max_seq_length

    @staticmethod
    def _pad(tensor: torch.Tensor, padding_length: int) -> torch.Tensor:
        batch_size, _, num_labels = tensor.shape
        padding = torch.zeros((batch_size, padding_length, num_labels), dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding], dim=1)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        ret: Dict[str, torch.Tensor] = self(batch)
        kdr_loss = compute_token_mean_loss(ret["kdr_logits"], batch["kdr_labels"])
        self.log("train/kdr_loss", kdr_loss)
        ins_loss = compute_token_mean_loss(ret["ins_logits"], batch["ins_labels"])
        self.log("train/ins_loss", ins_loss)
        return kdr_loss + ins_loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> None:
        kwargs = self.predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        kwargs.update({"kdr_labels": batch["kdr_labels"], "ins_labels": batch["ins_labels"]})
        corpus = self.valid_corpora[dataloader_idx or 0]
        self.valid_corpus2typo_module_metric[corpus].update(kwargs)

    def validation_epoch_end(self, validation_step_outputs) -> None:
        metrics_log: Dict[str, Dict[str, float]] = {corpus: {} for corpus in self.valid_corpora}
        for corpus, typo_module_metric in self.valid_corpus2typo_module_metric.items():
            dataset = self.trainer.val_dataloaders[self.valid_corpora.index(corpus)].dataset
            typo_module_metric.set_properties({"dataset": dataset})
            metrics = typo_module_metric.compute()
            metrics_log[corpus] = metrics
            typo_module_metric.reset()

        for corpus, metrics in metrics_log.items():
            self.log_dict({f"valid_{corpus}/{key}": value for key, value in metrics.items()})
        for key in list(metrics_log.values())[0].keys():
            mean_score = mean(metrics_log[corpus][key] for corpus in self.valid_corpora if key in metrics_log[corpus])
            self.log(f"valid/{key}", mean_score)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> None:
        kwargs = self.predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        kwargs.update({"kdr_labels": batch["kdr_labels"], "ins_labels": batch["ins_labels"]})
        corpus = self.test_corpora[dataloader_idx or 0]
        self.test_corpus2typo_module_metric[corpus].update(kwargs)

    def test_epoch_end(self, test_step_outputs) -> None:
        metrics_log: Dict[str, Dict[str, float]] = {corpus: {} for corpus in self.test_corpora}
        for corpus, typo_module_metric in self.test_corpus2typo_module_metric.items():
            dataset = self.trainer.test_dataloaders[self.test_corpora.index(corpus)].dataset
            typo_module_metric.set_properties({"dataset": dataset})
            metrics = typo_module_metric.compute()
            metrics_log[corpus] = metrics
            typo_module_metric.reset()

        for corpus, metrics in metrics_log.items():
            self.log_dict({f"test_{corpus}/{key}": value for key, value in metrics.items()})
        for key in list(metrics_log.values())[0].keys():
            mean_score = mean(metrics_log[corpus][key] for corpus in self.test_corpora if key in metrics_log[corpus])
            self.log(f"test/{key}", mean_score)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        ret: Dict[str, torch.Tensor] = self(batch)
        kdr_probabilities = ret["kdr_logits"].softmax(dim=2)
        kdr_max_probabilities, kdr_predictions = kdr_probabilities.max(dim=2)
        ins_probabilities = ret["ins_logits"].softmax(dim=2)
        ins_max_probabilities, ins_predictions = ins_probabilities.max(dim=2)
        return {
            "example_ids": batch["example_ids"],
            "kdr_predictions": kdr_predictions,
            "kdr_probabilities": kdr_max_probabilities,
            "ins_predictions": ins_predictions,
            "ins_probabilities": ins_max_probabilities,
        }
