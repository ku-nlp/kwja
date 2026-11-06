import os
from statistics import mean
from typing import TYPE_CHECKING, Any, cast

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedModel

from kwja.modules.base import BaseModule
from kwja.modules.components.head import SequenceLabelingHead
from kwja.modules.functions.loss import compute_token_mean_loss
from kwja.utils.constants import SENT_SEGMENTATION_TAGS, WORD_NORM_OP_TAGS, WORD_SEGMENTATION_TAGS

if TYPE_CHECKING:
    from kwja.metrics import CharModuleMetric
elif os.environ.get("KWJA_CLI_MODE") == "1":
    from kwja.modules.base import DummyModuleMetric as CharModuleMetric  # dummy class for faster loading
else:
    from kwja.metrics import CharModuleMetric


class CharModule(BaseModule[CharModuleMetric]):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__(hparams, CharModuleMetric(hparams.max_seq_length))

        self.encoder: PreTrainedModel = hydra.utils.call(
            hparams.encoder.from_config,
            _target_="kwja.modules.components.deberta_v2.DebertaV2Model._from_config",
            dtype=torch.float32,
        )
        if hasattr(hparams, "special_tokens"):
            self.encoder.resize_token_embeddings(self.encoder.config.vocab_size + len(hparams.special_tokens))
        head_kwargs: dict[str, Any] = dict(hidden_size=self.encoder.config.hidden_size, hidden_dropout_prob=0.05)

        # ---------- sentence segmentation ----------
        self.sent_segmentation_tagger = SequenceLabelingHead(len(SENT_SEGMENTATION_TAGS), **head_kwargs)

        # ---------- word segmentation ----------
        self.word_segmentation_tagger = SequenceLabelingHead(len(WORD_SEGMENTATION_TAGS), **head_kwargs)

        # ---------- word normalization ----------
        self.word_norm_op_tagger = SequenceLabelingHead(len(WORD_NORM_OP_TAGS), **head_kwargs)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.encoder = hydra.utils.call(
                self.hparams.encoder.from_pretrained,
                _target_="kwja.modules.components.deberta_v2.DebertaV2Model.from_pretrained",
            )
            if hasattr(self.hparams, "special_tokens"):
                special_tokens = cast(list[str], self.hparams.special_tokens)
                self.encoder.resize_token_embeddings(self.encoder.config.vocab_size + len(special_tokens))

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        encoded = self.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        return {
            "sent_segmentation_logits": self.sent_segmentation_tagger(encoded.last_hidden_state),
            "word_segmentation_logits": self.word_segmentation_tagger(encoded.last_hidden_state),
            "word_norm_op_logits": self.word_norm_op_tagger(encoded.last_hidden_state),
        }

    def training_step(self, batch: Any) -> torch.Tensor:
        ret: dict[str, torch.Tensor] = self(batch)
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

    def _get_metric_inputs(self, batch: Any) -> dict[str, torch.Tensor]:
        return {**self.predict_step(batch), "word_norm_op_labels": batch["word_norm_op_labels"]}

    def _compute_metrics(self, corpus: str, metric: CharModuleMetric, dataset: Dataset) -> dict[str, float]:
        metrics = super()._compute_metrics(corpus, metric, dataset)
        metrics["aggregated_char_metrics"] = mean(
            metrics[key] for key in self.hparams.aggregating_metrics if key in metrics
        )
        return metrics

    def predict_step(self, batch: Any) -> dict[str, torch.Tensor]:
        ret: dict[str, torch.Tensor] = self(batch)
        return {
            "example_ids": batch["example_ids"],
            "sent_segmentation_predictions": ret["sent_segmentation_logits"].argmax(dim=2),
            "word_segmentation_predictions": ret["word_segmentation_logits"].argmax(dim=2),
            "word_norm_op_predictions": ret["word_norm_op_logits"].argmax(dim=2),
        }
