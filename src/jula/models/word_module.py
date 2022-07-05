from statistics import mean
from typing import Any, Optional

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning.core.lightning import LightningModule
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from jula.evaluators.cohesion_analysis_metric import CohesionAnalysisMetric
from jula.evaluators.dependency_parsing_metric import DependencyParsingMetric
from jula.evaluators.phrase_analysis_metric import PhraseAnalysisMetric
from jula.evaluators.word_analyzer import WordAnalysisMetric
from jula.models.models.phrase_analyzer import PhraseAnalyzer
from jula.models.models.pooling import PoolingStrategy
from jula.models.models.relation_analyzer import RelationAnalyzer
from jula.models.models.word_analyzer import WordAnalyzer
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

        self.word_analyzer: WordAnalyzer = WordAnalyzer(
            pretrained_model_config=self.word_encoder.pretrained_model.config
        )
        self.valid_word_analysis_metrics: dict[str, WordAnalysisMetric] = {
            corpus: WordAnalysisMetric() for corpus in self.valid_corpora
        }
        self.test_word_analysis_metrics: dict[str, WordAnalysisMetric] = {
            corpus: WordAnalysisMetric() for corpus in self.test_corpora
        }

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

        self.relation_analyzer: RelationAnalyzer = RelationAnalyzer(
            hparams=hparams,
            pretrained_model_config=self.word_encoder.pretrained_model.config,
        )
        self.valid_dependency_parsing_metrics: dict[str, DependencyParsingMetric] = {
            corpus: DependencyParsingMetric() for corpus in self.valid_corpora
        }
        self.test_dependency_parsing_metrics: dict[str, DependencyParsingMetric] = {
            corpus: DependencyParsingMetric() for corpus in self.test_corpora
        }
        self.valid_cohesion_analysis_metrics: dict[str, CohesionAnalysisMetric] = {
            corpus: CohesionAnalysisMetric() for corpus in self.valid_corpora
        }
        self.test_cohesion_analysis_metrics: dict[str, CohesionAnalysisMetric] = {
            corpus: CohesionAnalysisMetric() for corpus in self.test_corpora
        }

    def forward(self, inference=False, **batch) -> dict[str, dict[str, torch.Tensor]]:
        # (batch_size, seq_len, hidden_size)
        pooled_outputs = self.word_encoder(batch, PoolingStrategy.FIRST)
        word_analyzer_outputs = self.word_analyzer(pooled_outputs, batch)
        phrase_analyzer_outputs = self.phrase_analyzer(pooled_outputs, batch)
        relation_analyzer_output = self.relation_analyzer(
            pooled_outputs, batch, inference=inference
        )
        return {
            "word_analyzer_outputs": word_analyzer_outputs,
            "phrase_analyzer_outputs": phrase_analyzer_outputs,
            "relation_analyzer_outputs": relation_analyzer_output,
        }

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs: dict[str, torch.Tensor] = self(inference=False, **batch)
        word_analysis_loss = outputs["word_analyzer_outputs"]["loss"]
        self.log(
            "train/word_analysis_loss",
            word_analysis_loss,
            on_step=True,
            on_epoch=True,
        )
        phrase_analysis_loss = outputs["phrase_analyzer_outputs"][
            "phrase_analysis_loss"
        ]
        self.log(
            "train/phrase_analysis_loss",
            phrase_analysis_loss,
            on_step=True,
            on_epoch=True,
        )
        dependency_loss = outputs["relation_analyzer_outputs"]["dependency_loss"]
        self.log("train/dependency_loss", dependency_loss, on_step=True, on_epoch=True)
        dependency_type_loss = outputs["relation_analyzer_outputs"][
            "dependency_type_loss"
        ]
        self.log(
            "train/dependency_type_loss",
            dependency_type_loss,
            on_step=True,
            on_epoch=True,
        )
        return (
            word_analysis_loss
            + phrase_analysis_loss
            + dependency_loss
            + dependency_type_loss
        )

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> None:
        outputs: dict[str, torch.Tensor] = self(inference=True, **batch)
        corpus = self.valid_corpora[dataloader_idx or 0]
        word_analysis_metric_args = {
            "pos_preds": torch.argmax(
                outputs["word_analyzer_outputs"]["pos_logits"], dim=-1
            ),
            "pos_labels": batch["mrph_types"][:, :, 0],
            "subpos_preds": torch.argmax(
                outputs["word_analyzer_outputs"]["subpos_logits"], dim=-1
            ),
            "subpos_labels": batch["mrph_types"][:, :, 1],
            "conjtype_preds": torch.argmax(
                outputs["word_analyzer_outputs"]["conjtype_logits"], dim=-1
            ),
            "conjtype_labels": batch["mrph_types"][:, :, 2],
            "conjform_preds": torch.argmax(
                outputs["word_analyzer_outputs"]["conjform_logits"], dim=-1
            ),
            "conjform_labels": batch["mrph_types"][:, :, 3],
        }
        self.valid_word_analysis_metrics[corpus].update(**word_analysis_metric_args)
        self.log(
            "valid/word_analysis_loss",
            outputs["word_analyzer_outputs"]["loss"],
        )

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

        dependency_parsing_metric_args = {
            "document_ids": batch["document_id"],
            "preds": torch.argmax(
                outputs["relation_analyzer_outputs"]["dependency_logits"], dim=2
            ),
            "type_preds": torch.argmax(
                outputs["relation_analyzer_outputs"]["dependency_type_logits"], dim=2
            ),
        }
        self.valid_dependency_parsing_metrics[corpus].update(
            **dependency_parsing_metric_args
        )
        self.log(
            "valid/dependency_loss",
            outputs["relation_analyzer_outputs"]["dependency_loss"],
        )

        cohesion_analysis_metric_args = {
            "example_ids": batch["document_id"],
            "output": outputs["relation_analyzer_outputs"]["cohesion_logits"],
            "dataset": self.trainer.val_dataloaders[dataloader_idx or 0].dataset,
        }
        self.valid_cohesion_analysis_metrics[corpus].update(
            **cohesion_analysis_metric_args
        )
        self.log(
            "valid/cohesion_loss",
            outputs["relation_analyzer_outputs"]["cohesion_loss"],
        )

    def validation_epoch_end(self, validation_step_outputs) -> None:
        f1s: list[float] = []
        word_analysis_f1 = 0.0
        for corpus, metric in self.valid_word_analysis_metrics.items():
            for name, value in metric.compute().items():
                if name == "word_analysis_f1":
                    word_analysis_f1 += value
                self.log(f"valid_{corpus}/{name}", value)
                metric.reset()
        self.log(
            "valid/word_analysis_f1",
            word_analysis_f1 / len(self.valid_word_analysis_metrics),
        )
        f1s.append(word_analysis_f1 / len(self.valid_word_analysis_metrics))

        phrase_analysis_f1 = 0.0
        for corpus, metric in self.valid_phrase_analysis_metrics.items():
            for name, value in metric.compute().items():
                if name == "phrase_analysis_f1":
                    phrase_analysis_f1 += value
                self.log(f"valid_{corpus}/{name}", value)
            metric.reset()
        self.log(
            "valid/phrase_analysis_f1",
            phrase_analysis_f1 / len(self.valid_phrase_analysis_metrics),
        )
        f1s.append(phrase_analysis_f1 / len(self.valid_phrase_analysis_metrics))

        self.log("valid/f1", mean(f1s))

        for corpus, metric in self.valid_dependency_parsing_metrics.items():
            dataset = self.trainer.datamodule.valid_datasets[corpus]
            for name, value in metric.compute(dataset).items():
                self.log(f"valid_{corpus}/{name}", value)
            metric.reset()

        for corpus, metric in self.valid_cohesion_analysis_metrics.items():
            dataset = self.trainer.datamodule.valid_datasets[corpus]
            for rel, val in metric.compute(dataset).to_dict().items():
                for met, sub_val in val.items():
                    self.log(f"valid_{corpus}/{met}_{rel}", sub_val.f1)
            metric.reset()

    def test_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> None:
        outputs: dict[str, torch.Tensor] = self(inference=True, **batch)
        corpus = self.test_corpora[dataloader_idx or 0]
        word_analysis_metric_args = {
            "pos_preds": torch.argmax(
                outputs["word_analyzer_outputs"]["pos_logits"], dim=-1
            ),
            "pos_labels": batch["mrph_types"][:, :, 0],
            "subpos_preds": torch.argmax(
                outputs["word_analyzer_outputs"]["subpos_logits"], dim=-1
            ),
            "subpos_labels": batch["mrph_types"][:, :, 1],
            "conjtype_preds": torch.argmax(
                outputs["word_analyzer_outputs"]["conjtype_logits"], dim=-1
            ),
            "conjtype_labels": batch["mrph_types"][:, :, 2],
            "conjform_preds": torch.argmax(
                outputs["word_analyzer_outputs"]["conjform_logits"], dim=-1
            ),
            "conjform_labels": batch["mrph_types"][:, :, 3],
        }
        self.test_word_analysis_metrics[corpus].update(**word_analysis_metric_args)
        self.log(
            "test/word_analysis_loss",
            outputs["word_analyzer_outputs"]["loss"],
        )
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

        dependency_parsing_metric_args = {
            "document_ids": batch["document_id"],
            "preds": torch.argmax(
                outputs["relation_analyzer_outputs"]["dependency_logits"], dim=2
            ),
            "type_preds": torch.argmax(
                outputs["relation_analyzer_outputs"]["dependency_type_logits"], dim=2
            ),
        }
        self.test_dependency_parsing_metrics[corpus].update(
            **dependency_parsing_metric_args
        )
        self.log(
            "test/dependency_loss",
            outputs["relation_analyzer_outputs"]["dependency_loss"],
        )

        cohesion_analysis_metric_args = {
            "example_ids": batch["document_id"],
            "output": outputs["relation_analyzer_outputs"]["cohesion_logits"],
            "dataset": self.trainer.test_dataloaders[dataloader_idx or 0].dataset,
        }
        self.test_cohesion_analysis_metrics[corpus].update(
            **cohesion_analysis_metric_args
        )
        self.log(
            "test/cohesion_loss",
            outputs["relation_analyzer_outputs"]["cohesion_loss"],
        )

    def test_epoch_end(self, test_step_outputs) -> None:
        f1s: list[float] = []
        word_analysis_f1 = 0.0
        for corpus, metric in self.test_word_analysis_metrics.items():
            for name, value in metric.compute().items():
                if name == "word_analysis_f1":
                    word_analysis_f1 += value
                self.log(f"test_{corpus}/{name}", value)
                metric.reset()
        self.log(
            "test/word_analysis_f1",
            word_analysis_f1 / len(self.test_word_analysis_metrics),
        )
        f1s.append(word_analysis_f1 / len(self.test_word_analysis_metrics))

        phrase_analysis_f1 = 0.0
        for corpus, metric in self.test_phrase_analysis_metrics.items():
            for name, value in metric.compute().items():
                if name == "phrase_analysis_f1":
                    phrase_analysis_f1 += value
                self.log(f"test_{corpus}/{name}", value)
            metric.reset()
        self.log(
            "test/phrase_analysis_f1",
            phrase_analysis_f1 / len(self.test_phrase_analysis_metrics),
        )
        f1s.append(phrase_analysis_f1 / len(self.test_phrase_analysis_metrics))

        self.log("test/f1", mean(f1s))

        for corpus, metric in self.test_dependency_parsing_metrics.items():
            dataset = self.trainer.datamodule.test_datasets[corpus]
            for name, value in metric.compute(dataset).items():
                self.log(f"test_{corpus}/{name}", value)
            metric.reset()

        for corpus, metric in self.test_cohesion_analysis_metrics.items():
            dataset = self.trainer.datamodule.test_datasets[corpus]
            for rel, val in metric.compute(dataset).to_dict().items():
                for met, sub_val in val.items():
                    self.log(f"test_{corpus}/{met}_{rel}", sub_val.f1)
            metric.reset()

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        outputs: dict[str, torch.Tensor] = self(inference=True, **batch)
        return {
            "document_ids": batch["document_id"],
            "word_analysis_pos_logits": outputs["word_analyzer_outputs"]["pos_logits"],
            "word_analysis_subpos_logits": outputs["word_analyzer_outputs"][
                "subpos_logits"
            ],
            "word_analysis_conjtype_logits": outputs["word_analyzer_outputs"][
                "conjtype_logits"
            ],
            "word_analysis_conjform_logits": outputs["word_analyzer_outputs"][
                "conjform_logits"
            ],
            "phrase_analysis_logits": outputs["phrase_analyzer_outputs"][
                "phrase_analysis_logits"
            ],
            # "base_phrase_features": batch["base_phrase_features"],
            "dependency_logits": outputs["relation_analyzer_outputs"][
                "dependency_logits"
            ],
            "dependency_type_logits": outputs["relation_analyzer_outputs"][
                "dependency_type_logits"
            ],
        }

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, params=self.parameters(), _convert_="partial"
        )
        return [optimizer], []
