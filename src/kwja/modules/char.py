from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
from omegaconf import DictConfig
from transformers import PretrainedConfig, PreTrainedModel

from kwja.metrics import CharModuleMetric
from kwja.modules.base import BaseModule
from kwja.modules.components.head import SequenceLabelingHead
from kwja.modules.functions.loss import compute_token_mean_loss
from kwja.utils.constants import WORD_NORM_OP_TAGS, WORD_SEGMENTATION_TAGS


class CharModule(BaseModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__(hparams)

        if valid_corpora := getattr(hparams.datamodule, "valid", None):
            self.valid_corpora: List[str] = list(valid_corpora)
            self.valid_corpus2char_module_metric: Dict[str, CharModuleMetric] = {
                corpus: CharModuleMetric() for corpus in self.valid_corpora
            }
        if test_corpora := getattr(hparams.datamodule, "test", None):
            self.test_corpora: List[str] = list(test_corpora)
            self.test_corpus2char_module_metric: Dict[str, CharModuleMetric] = {
                corpus: CharModuleMetric() for corpus in self.test_corpora
            }

        self.char_encoder: PreTrainedModel = hydra.utils.call(hparams.encoder)
        pretrained_model_config: PretrainedConfig = self.char_encoder.config
        if hasattr(hparams, "special_tokens"):
            self.char_encoder.resize_token_embeddings(pretrained_model_config.vocab_size + len(hparams.special_tokens))
        head_args: Tuple[int, float] = (
            pretrained_model_config.hidden_size,
            pretrained_model_config.hidden_dropout_prob,
        )

        # ---------- word segmentation ----------
        self.word_segmentation_tagger = SequenceLabelingHead(len(WORD_SEGMENTATION_TAGS), *head_args)

        # ---------- word normalization ----------
        self.word_norm_op_tagger = SequenceLabelingHead(len(WORD_NORM_OP_TAGS), *head_args)

    def forward(self, batch: Any) -> Dict[str, torch.Tensor]:
        truncation_length = self._truncate(batch)
        encoded = self.char_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        word_segmentation_logits = self.word_segmentation_tagger(encoded.last_hidden_state)
        word_norm_op_logits = self.word_norm_op_tagger(encoded.last_hidden_state)
        if truncation_length > 0:
            word_segmentation_logits = self._pad(word_segmentation_logits, truncation_length)
            word_norm_op_logits = self._pad(word_norm_op_logits, truncation_length)
        return {
            "word_segmentation_logits": word_segmentation_logits,
            "word_norm_op_logits": word_norm_op_logits,
        }

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
        word_segmentation_loss = compute_token_mean_loss(
            ret["word_segmentation_logits"], batch["word_segmentation_labels"]
        )
        self.log("train/word_segmentation_loss", word_segmentation_loss)
        word_normalization_loss = compute_token_mean_loss(ret["word_norm_op_logits"], batch["word_norm_op_labels"])
        self.log("train/word_normalization_loss", word_normalization_loss)
        return word_segmentation_loss + word_normalization_loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> None:
        kwargs = self.predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        kwargs.update({"word_norm_op_labels": batch["word_norm_op_labels"]})
        corpus = self.valid_corpora[dataloader_idx or 0]
        self.valid_corpus2char_module_metric[corpus].update(kwargs)

    def validation_epoch_end(self, validation_step_outputs) -> None:
        metrics_log: Dict[str, Dict[str, float]] = {corpus: {} for corpus in self.valid_corpora}
        for corpus, char_module_metric in self.valid_corpus2char_module_metric.items():
            dataset = self.trainer.val_dataloaders[self.valid_corpora.index(corpus)].dataset
            char_module_metric.set_properties({"dataset": dataset})
            metrics = char_module_metric.compute()
            metrics["aggregated_char_metrics"] = mean(
                metrics[key] for key in self.hparams.aggregating_metrics if key in metrics
            )
            metrics_log[corpus] = metrics
            char_module_metric.reset()

        for corpus, metrics in metrics_log.items():
            self.log_dict({f"valid_{corpus}/{key}": value for key, value in metrics.items()})
        for key in list(metrics_log.values())[0].keys():
            mean_score = mean(metrics_log[corpus][key] for corpus in self.valid_corpora if key in metrics_log[corpus])
            self.log(f"valid/{key}", mean_score)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> None:
        kwargs = self.predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        kwargs.update({"word_norm_op_labels": batch["word_norm_op_labels"]})
        corpus = self.test_corpora[dataloader_idx or 0]
        self.test_corpus2char_module_metric[corpus].update(kwargs)

    def test_epoch_end(self, test_step_outputs) -> None:
        metrics_log: Dict[str, Dict[str, float]] = {corpus: {} for corpus in self.test_corpora}
        for corpus, char_module_metric in self.test_corpus2char_module_metric.items():
            dataset = self.trainer.test_dataloaders[self.test_corpora.index(corpus)].dataset
            char_module_metric.set_properties({"dataset": dataset})
            metrics = char_module_metric.compute()
            metrics["aggregated_char_metrics"] = mean(
                metrics[key] for key in self.hparams.aggregating_metrics if key in metrics
            )
            metrics_log[corpus] = metrics
            char_module_metric.reset()

        for corpus, metrics in metrics_log.items():
            self.log_dict({f"test_{corpus}/{key}": value for key, value in metrics.items()})
        for key in list(metrics_log.values())[0].keys():
            mean_score = mean(metrics_log[corpus][key] for corpus in self.test_corpora if key in metrics_log[corpus])
            self.log(f"test/{key}", mean_score)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        ret: Dict[str, torch.Tensor] = self(batch)
        return {
            "example_ids": batch["example_ids"],
            "word_segmentation_predictions": ret["word_segmentation_logits"].argmax(dim=2),
            "word_norm_op_predictions": ret["word_norm_op_logits"].argmax(dim=2),
        }
