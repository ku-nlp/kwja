from typing import Any, Optional

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning.core.lightning import LightningModule
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from jula.evaluators.phrase_analysis_metric import PhraseAnalysisMetric
from jula.models.models.phrase_analyzer import PhraseAnalyzer
from jula.models.models.pooling import PoolingStrategy
from jula.models.models.word_encoder import WordEncoder


class WordModule(LightningModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            hparams.model.model_name_or_path,
            **hydra.utils.instantiate(
                hparams.dataset.tokenizer_kwargs, _convert_="partial"
            ),
        )

        self.valid_corpora = list(hparams.dataset.valid.keys())
        self.test_corpora = list(hparams.dataset.test.keys())

        self.word_encoder: WordEncoder = WordEncoder(hparams, self.tokenizer)

        self.phrase_analyzer: PhraseAnalyzer = PhraseAnalyzer(
            hparams=hparams,
            pretrained_model_config=self.word_encoder.pretrained_model.config,
        )
        self.valid_phrase_analysis_metrics: dict[str, PhraseAnalysisMetric] = {
            corpus: PhraseAnalysisMetric() for corpus in self.valid_corpora
        }
        self.test_phrase_analysis_metrics: dict[str, PhraseAnalysisMetric] = {
            corpus: PhraseAnalysisMetric() for corpus in self.test_corpora
        }

    def forward(self, **batch) -> dict[str, dict[str, torch.Tensor]]:
        # (batch_size, seq_len, hidden_size)
        pooled_outputs = self.word_encoder(batch, PoolingStrategy.FIRST)
        phrase_analyzer_outputs = self.phrase_analyzer(pooled_outputs, batch)
        return {"phrase_analyzer_outputs": phrase_analyzer_outputs}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs: dict[str, torch.Tensor] = self(**batch)
        loss = outputs["phrase_analyzer_outputs"]["phrase_analysis_loss"]
        self.log("train/phrase_analysis_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> None:
        outputs: dict[str, torch.Tensor] = self(**batch)
        corpus = self.valid_corpora[dataloader_idx or 0]
        phrase_analysis_metric_args = {
            "document_ids": batch["document_id"],
            "preds": torch.where(
                outputs["phrase_analyzer_outputs"]["phrase_analysis_logits"] >= 0.5,
                1.0,
                0.0,
            ),
            "labels": batch["base_phrase_features"],
        }
        self.valid_phrase_analysis_metrics[corpus].update(**phrase_analysis_metric_args)
        self.log(
            "valid/phrase_analysis_loss",
            outputs["phrase_analyzer_outputs"]["phrase_analysis_loss"],
        )

    def validation_epoch_end(self, validation_step_outputs) -> None:
        f1 = 0.0
        for corpus, metric in self.valid_phrase_analysis_metrics.items():
            for name, value in metric.compute().items():
                if name == "phrase_analysis_f1":
                    f1 += value
                self.log(f"valid_{corpus}/{name}", value)
            metric.reset()
        self.log("valid/f1", f1 / len(self.valid_phrase_analysis_metrics))

    def test_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> None:
        outputs: dict[str, torch.Tensor] = self(**batch)
        corpus = self.test_corpora[dataloader_idx or 0]
        phrase_analysis_metric_args = {
            "document_ids": batch["document_id"],
            "preds": torch.where(
                outputs["phrase_analyzer_outputs"]["phrase_analysis_logits"] >= 0.5,
                1.0,
                0.0,
            ),
            "labels": batch["base_phrase_features"],
        }
        self.test_phrase_analysis_metrics[corpus].update(**phrase_analysis_metric_args)
        self.log(
            "test/phrase_analysis_loss",
            outputs["phrase_analyzer_outputs"]["phrase_analysis_loss"],
        )

    def test_epoch_end(self, test_step_outputs) -> None:
        f1 = 0.0
        for corpus, metric in self.test_phrase_analysis_metrics.items():
            for name, value in metric.compute().items():
                if name == "phrase_analysis_f1":
                    f1 += value
                self.log(f"test_{corpus}/{name}", value)
            metric.reset()
        self.log("test/f1", f1 / len(self.valid_phrase_analysis_metrics))

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        outputs: dict[str, torch.Tensor] = self(**batch)
        return {
            "document_ids": batch["document_id"],
            "phrase_analysis_logits": outputs["phrase_analyzer_outputs"][
                "phrase_analysis_logits"
            ],
            "base_phrase_features": batch["base_phrase_features"],
        }

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, params=self.parameters(), _convert_="partial"
        )
        return [optimizer], []
