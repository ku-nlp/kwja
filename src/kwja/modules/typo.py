import os
from statistics import mean
from typing import Any, Dict

import hydra
import torch
from omegaconf import DictConfig
from transformers import PreTrainedModel

from kwja.modules.base import BaseModule
from kwja.modules.components.head import SequenceLabelingHead
from kwja.modules.functions.loss import compute_token_mean_loss
from kwja.utils.constants import RESOURCE_PATH

if os.environ.get("KWJA_CLI_MODE") == "1":
    from kwja.modules.base import DummyModuleMetric as TypoModuleMetric  # dummy class for faster loading
else:
    from kwja.metrics import TypoModuleMetric  # type: ignore


class TypoModule(BaseModule[TypoModuleMetric]):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__(hparams, TypoModuleMetric(hparams.max_seq_length))

        self.encoder: PreTrainedModel = hydra.utils.call(hparams.encoder.from_config)
        if hasattr(hparams, "special_tokens"):
            self.encoder.resize_token_embeddings(self.encoder.config.vocab_size + len(hparams.special_tokens))
        head_kwargs: Dict[str, Any] = dict(hidden_size=self.encoder.config.hidden_size, hidden_dropout_prob=0.05)

        self.kdr_tagger = SequenceLabelingHead(self.encoder.config.vocab_size, **head_kwargs)
        extended_vocab_size = sum(1 for _ in RESOURCE_PATH.joinpath("typo_correction", "multi_char_vocab.txt").open())
        self.ins_tagger = SequenceLabelingHead(self.encoder.config.vocab_size + extended_vocab_size, **head_kwargs)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.encoder = hydra.utils.call(self.hparams.encoder.from_pretrained)
            if hasattr(self.hparams, "special_tokens"):
                self.encoder.resize_token_embeddings(self.encoder.config.vocab_size + len(self.hparams.special_tokens))

    def forward(self, batch: Any) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        return {
            "kdr_logits": self.kdr_tagger(encoded.last_hidden_state),
            "ins_logits": self.ins_tagger(encoded.last_hidden_state),
        }

    def training_step(self, batch: Any) -> torch.Tensor:
        ret: Dict[str, torch.Tensor] = self(batch)
        kdr_loss = compute_token_mean_loss(ret["kdr_logits"], batch["kdr_labels"])
        self.log("train/kdr_loss", kdr_loss)
        ins_loss = compute_token_mean_loss(ret["ins_logits"], batch["ins_labels"])
        self.log("train/ins_loss", ins_loss)
        return kdr_loss + ins_loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        kwargs = self.predict_step(batch)
        kwargs.update({"kdr_labels": batch["kdr_labels"], "ins_labels": batch["ins_labels"]})
        metric = self.valid_corpus2metric[self.valid_corpora[dataloader_idx]]
        metric.update(kwargs)

    def on_validation_epoch_end(self) -> None:
        metrics_log: Dict[str, Dict[str, float]] = {}
        for corpus, metric in self.valid_corpus2metric.items():
            dataset = self.trainer.val_dataloaders[corpus].dataset
            metric.set_properties({"dataset": dataset})
            metrics_log[corpus] = metric.compute()
            metric.reset()

        for corpus, metrics in metrics_log.items():
            self.log_dict({f"valid_{corpus}/{key}": value for key, value in metrics.items()})
        for key in list(metrics_log.values())[0].keys():
            mean_score = mean(metrics_log[corpus][key] for corpus in self.valid_corpora if key in metrics_log[corpus])
            self.log(f"valid/{key}", mean_score)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        kwargs = self.predict_step(batch)
        kwargs.update({"kdr_labels": batch["kdr_labels"], "ins_labels": batch["ins_labels"]})
        metric = self.test_corpus2metric[self.test_corpora[dataloader_idx]]
        metric.update(kwargs)

    def on_test_epoch_end(self) -> None:
        metrics_log: Dict[str, Dict[str, float]] = {}
        for corpus, metric in self.test_corpus2metric.items():
            dataset = self.trainer.test_dataloaders[corpus].dataset
            metric.set_properties({"dataset": dataset})
            metrics_log[corpus] = metric.compute()
            metric.reset()

        for corpus, metrics in metrics_log.items():
            self.log_dict({f"test_{corpus}/{key}": value for key, value in metrics.items()})
        for key in list(metrics_log.values())[0].keys():
            mean_score = mean(metrics_log[corpus][key] for corpus in self.test_corpora if key in metrics_log[corpus])
            self.log(f"test/{key}", mean_score)

    def predict_step(self, batch: Any) -> Dict[str, torch.Tensor]:
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
