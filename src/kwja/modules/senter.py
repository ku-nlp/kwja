from statistics import mean
from typing import Any, Dict, List, Optional

import hydra
import torch
from omegaconf import DictConfig
from transformers import PretrainedConfig, PreTrainedModel

from kwja.metrics import SenterModuleMetric
from kwja.modules.base import BaseModule
from kwja.modules.components.head import SequenceLabelingHead
from kwja.modules.functions.loss import compute_token_mean_loss
from kwja.utils.constants import SENT_SEGMENTATION_TAGS


class SenterModule(BaseModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__(hparams)

        if valid_corpora := getattr(hparams.datamodule, "valid", None):
            self.valid_corpora: List[str] = list(valid_corpora)
            self.valid_corpus2senter_module_metric: Dict[str, SenterModuleMetric] = {
                corpus: SenterModuleMetric() for corpus in self.valid_corpora
            }
        if test_corpora := getattr(hparams.datamodule, "test", None):
            self.test_corpora: List[str] = list(test_corpora)
            self.test_corpus2senter_module_metric: Dict[str, SenterModuleMetric] = {
                corpus: SenterModuleMetric() for corpus in self.test_corpora
            }

        self.char_encoder: PreTrainedModel = hydra.utils.call(hparams.encoder)
        pretrained_model_config: PretrainedConfig = self.char_encoder.config
        if hasattr(hparams, "special_tokens"):
            self.char_encoder.resize_token_embeddings(pretrained_model_config.vocab_size + len(hparams.special_tokens))
        self.sent_segmentation_tagger = SequenceLabelingHead(
            num_labels=len(SENT_SEGMENTATION_TAGS),
            hidden_size=pretrained_model_config.hidden_size,
            hidden_dropout_prob=pretrained_model_config.hidden_dropout_prob,
        )

    def forward(self, batch: Any) -> Dict[str, Dict[str, torch.Tensor]]:
        encoded = self.char_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        sent_segmentation_logits = self.sent_segmentation_tagger(encoded.last_hidden_state)
        return {"sent_segmentation_logits": sent_segmentation_logits}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        ret: Dict[str, torch.Tensor] = self(batch)
        sent_segmentation_loss = compute_token_mean_loss(
            ret["sent_segmentation_logits"], batch["sent_segmentation_labels"]
        )
        self.log("train/sent_segmentation_loss", sent_segmentation_loss)
        return sent_segmentation_loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        kwargs = self.predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        corpus = self.valid_corpora[dataloader_idx or 0]
        self.valid_corpus2senter_module_metric[corpus].update(kwargs)

    def validation_epoch_end(self, validation_step_outputs) -> None:
        metrics_log: Dict[str, Dict[str, float]] = {corpus: {} for corpus in self.valid_corpora}
        for corpus, senter_module_metric in self.valid_corpus2senter_module_metric.items():
            dataset = self.trainer.val_dataloaders[self.valid_corpora.index(corpus)].dataset
            senter_module_metric.set_properties({"dataset": dataset})
            metrics = senter_module_metric.compute()
            metrics["aggregated_senter_metrics"] = mean(
                metrics[key] for key in self.hparams.aggregating_metrics if key in metrics
            )
            metrics_log[corpus] = metrics
            senter_module_metric.reset()

        for corpus, metrics in metrics_log.items():
            self.log_dict({f"valid_{corpus}/{key}": value for key, value in metrics.items()})
        for key in list(metrics_log.values())[0].keys():
            mean_score = mean(metrics_log[corpus][key] for corpus in self.valid_corpora if key in metrics_log[corpus])
            self.log(f"valid/{key}", mean_score)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        kwargs = self.predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        corpus = self.test_corpora[dataloader_idx or 0]
        self.test_corpus2senter_module_metric[corpus].update(kwargs)

    def test_epoch_end(self, test_step_outputs) -> None:
        metrics_log: Dict[str, Dict[str, float]] = {corpus: {} for corpus in self.test_corpora}
        for corpus, senter_module_metric in self.test_corpus2senter_module_metric.items():
            dataset = self.trainer.test_dataloaders[self.test_corpora.index(corpus)].dataset
            senter_module_metric.set_properties({"dataset": dataset})
            metrics = senter_module_metric.compute()
            metrics["aggregated_senter_metrics"] = mean(
                metrics[key] for key in self.hparams.aggregating_metrics if key in metrics
            )
            metrics_log[corpus] = metrics
            senter_module_metric.reset()

        for corpus, metrics in metrics_log.items():
            self.log_dict({f"test_{corpus}/{key}": value for key, value in metrics.items()})
        for key in list(metrics_log.values())[0].keys():
            mean_score = mean(metrics_log[corpus][key] for corpus in self.test_corpora if key in metrics_log[corpus])
            self.log(f"test/{key}", mean_score)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        ret: Dict[str, torch.Tensor] = self(batch)
        return {
            "example_ids": batch["example_ids"],
            "sent_segmentation_predictions": ret["sent_segmentation_logits"].argmax(dim=2),
        }
