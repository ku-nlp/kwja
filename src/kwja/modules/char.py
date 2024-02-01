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
from kwja.utils.constants import SENT_SEGMENTATION_TAGS, WORD_NORM_OP_TAGS, WORD_SEGMENTATION_TAGS

if os.environ.get("KWJA_CLI_MODE") == "1":
    from kwja.modules.base import DummyModuleMetric as CharModuleMetric  # dummy class for faster loading
else:
    from kwja.metrics import CharModuleMetric  # type: ignore


class CharModule(BaseModule[CharModuleMetric]):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__(hparams, CharModuleMetric(hparams.max_seq_length))

        self.encoder: PreTrainedModel = hydra.utils.call(hparams.encoder.from_config)
        if hasattr(hparams, "special_tokens"):
            self.encoder.resize_token_embeddings(self.encoder.config.vocab_size + len(hparams.special_tokens))
        head_kwargs: Dict[str, Any] = dict(hidden_size=self.encoder.config.hidden_size, hidden_dropout_prob=0.05)

        # ---------- sentence segmentation ----------
        self.sent_segmentation_tagger = SequenceLabelingHead(len(SENT_SEGMENTATION_TAGS), **head_kwargs)

        # ---------- word segmentation ----------
        self.word_segmentation_tagger = SequenceLabelingHead(len(WORD_SEGMENTATION_TAGS), **head_kwargs)

        # ---------- word normalization ----------
        self.word_norm_op_tagger = SequenceLabelingHead(len(WORD_NORM_OP_TAGS), **head_kwargs)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.encoder = hydra.utils.call(self.hparams.encoder.from_pretrained)
            if hasattr(self.hparams, "special_tokens"):
                self.encoder.resize_token_embeddings(self.encoder.config.vocab_size + len(self.hparams.special_tokens))

    def forward(self, batch: Any) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        return {
            "sent_segmentation_logits": self.sent_segmentation_tagger(encoded.last_hidden_state),
            "word_segmentation_logits": self.word_segmentation_tagger(encoded.last_hidden_state),
            "word_norm_op_logits": self.word_norm_op_tagger(encoded.last_hidden_state),
        }

    def training_step(self, batch: Any) -> torch.Tensor:
        ret: Dict[str, torch.Tensor] = self(batch)
        sent_segmentation_loss = compute_token_mean_loss(
            ret["sent_segmentation_logits"], batch["sent_segmentation_labels"]
        )
        self.log("train/sent_segmentation_loss", sent_segmentation_loss)
        word_segmentation_loss = compute_token_mean_loss(
            ret["word_segmentation_logits"], batch["word_segmentation_labels"]
        )
        self.log("train/word_segmentation_loss", word_segmentation_loss)
        word_normalization_loss = compute_token_mean_loss(ret["word_norm_op_logits"], batch["word_norm_op_labels"])
        self.log("train/word_normalization_loss", word_normalization_loss)
        return sent_segmentation_loss + word_segmentation_loss + word_normalization_loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        kwargs = self.predict_step(batch)
        kwargs.update({"word_norm_op_labels": batch["word_norm_op_labels"]})
        metric = self.valid_corpus2metric[self.valid_corpora[dataloader_idx]]
        metric.update(kwargs)

    def on_validation_epoch_end(self) -> None:
        metrics_log: Dict[str, Dict[str, float]] = {}
        for corpus, metric in self.valid_corpus2metric.items():
            dataset = self.trainer.val_dataloaders[corpus].dataset
            metric.set_properties({"dataset": dataset})
            metrics = metric.compute()
            metrics["aggregated_char_metrics"] = mean(
                metrics[key] for key in self.hparams.aggregating_metrics if key in metrics
            )
            metrics_log[corpus] = metrics
            metric.reset()

        for corpus, metrics in metrics_log.items():
            self.log_dict({f"valid_{corpus}/{key}": value for key, value in metrics.items()})
        for key in list(metrics_log.values())[0].keys():
            mean_score = mean(metrics_log[corpus][key] for corpus in self.valid_corpora if key in metrics_log[corpus])
            self.log(f"valid/{key}", mean_score)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        kwargs = self.predict_step(batch)
        kwargs.update({"word_norm_op_labels": batch["word_norm_op_labels"]})
        metric = self.test_corpus2metric[self.test_corpora[dataloader_idx]]
        metric.update(kwargs)

    def on_test_epoch_end(self) -> None:
        metrics_log: Dict[str, Dict[str, float]] = {}
        for corpus, metric in self.test_corpus2metric.items():
            dataset = self.trainer.test_dataloaders[corpus].dataset
            metric.set_properties({"dataset": dataset})
            metrics = metric.compute()
            metrics["aggregated_char_metrics"] = mean(
                metrics[key] for key in self.hparams.aggregating_metrics if key in metrics
            )
            metrics_log[corpus] = metrics
            metric.reset()

        for corpus, metrics in metrics_log.items():
            self.log_dict({f"test_{corpus}/{key}": value for key, value in metrics.items()})
        for key in list(metrics_log.values())[0].keys():
            mean_score = mean(metrics_log[corpus][key] for corpus in self.test_corpora if key in metrics_log[corpus])
            self.log(f"test/{key}", mean_score)

    def predict_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        ret: Dict[str, torch.Tensor] = self(batch)
        return {
            "example_ids": batch["example_ids"],
            "sent_segmentation_predictions": ret["sent_segmentation_logits"].argmax(dim=2),
            "word_segmentation_predictions": ret["word_segmentation_logits"].argmax(dim=2),
            "word_norm_op_predictions": ret["word_norm_op_logits"].argmax(dim=2),
        }
