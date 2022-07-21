from collections import defaultdict
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
            **hydra.utils.instantiate(hparams.dataset.tokenizer_kwargs, _convert_="partial"),
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

    def forward(self, **batch) -> dict[str, dict[str, torch.Tensor]]:
        # (batch_size, seq_len, hidden_size)
        pooled_outputs = self.word_encoder(batch, PoolingStrategy.FIRST)
        word_analyzer_outputs = self.word_analyzer(pooled_outputs, batch)
        phrase_analyzer_outputs = self.phrase_analyzer(pooled_outputs, batch)
        relation_analyzer_output = self.relation_analyzer(pooled_outputs, batch)
        return {
            "word_analyzer_outputs": word_analyzer_outputs,
            "phrase_analyzer_outputs": phrase_analyzer_outputs,
            "relation_analyzer_outputs": relation_analyzer_output,
        }

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        batch["training"] = True
        outputs: dict[str, torch.Tensor] = self(**batch)
        word_analysis_loss = outputs["word_analyzer_outputs"]["loss"]
        self.log(
            "train/word_analysis_loss",
            word_analysis_loss,
            on_step=True,
            on_epoch=False,
        )
        word_feature_loss = outputs["phrase_analyzer_outputs"]["word_feature_loss"]
        self.log(
            "train/word_feature_loss",
            word_feature_loss,
            on_step=True,
            on_epoch=False,
        )
        base_phrase_feature_loss = outputs["phrase_analyzer_outputs"]["base_phrase_feature_loss"]
        self.log(
            "train/base_phrase_feature_loss",
            base_phrase_feature_loss,
            on_step=True,
            on_epoch=False,
        )
        dependency_loss = outputs["relation_analyzer_outputs"]["dependency_loss"]
        self.log("train/dependency_loss", dependency_loss, on_step=True, on_epoch=True)
        dependency_type_loss = outputs["relation_analyzer_outputs"]["dependency_type_loss"]
        self.log(
            "train/dependency_type_loss",
            dependency_type_loss,
            on_step=True,
            on_epoch=False,
        )
        cohesion_loss = outputs["relation_analyzer_outputs"]["cohesion_loss"]
        self.log(
            "train/cohesion_loss",
            cohesion_loss,
            on_step=True,
            on_epoch=False,
        )
        discourse_parsing_loss = outputs["relation_analyzer_outputs"]["discourse_parsing_loss"]
        self.log(
            "train/discourse_parsing_loss",
            discourse_parsing_loss,
            on_step=True,
            on_epoch=False,
        )
        return (
            word_analysis_loss
            + word_feature_loss
            + base_phrase_feature_loss
            + dependency_loss
            + dependency_type_loss
            + cohesion_loss
            + discourse_parsing_loss
        )

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> None:
        batch["training"] = False
        outputs: dict[str, torch.Tensor] = self(**batch)
        corpus = self.valid_corpora[dataloader_idx or 0]
        word_analysis_metric_args = {
            "pos_preds": torch.argmax(outputs["word_analyzer_outputs"]["pos_logits"], dim=-1),
            "pos_labels": batch["mrph_types"][:, :, 0],
            "subpos_preds": torch.argmax(outputs["word_analyzer_outputs"]["subpos_logits"], dim=-1),
            "subpos_labels": batch["mrph_types"][:, :, 1],
            "conjtype_preds": torch.argmax(outputs["word_analyzer_outputs"]["conjtype_logits"], dim=-1),
            "conjtype_labels": batch["mrph_types"][:, :, 2],
            "conjform_preds": torch.argmax(outputs["word_analyzer_outputs"]["conjform_logits"], dim=-1),
            "conjform_labels": batch["mrph_types"][:, :, 3],
        }
        self.valid_word_analysis_metrics[corpus].update(**word_analysis_metric_args)
        self.log(
            "valid/word_analysis_loss",
            outputs["word_analyzer_outputs"]["loss"],
        )

        phrase_analysis_metric_args = {
            "example_ids": batch["example_ids"],
            "word_feature_predictions": outputs["phrase_analyzer_outputs"]["word_feature_logits"].ge(0.5).long(),
            "word_features": batch["word_features"],
            "base_phrase_feature_predictions": outputs["phrase_analyzer_outputs"]["base_phrase_feature_logits"]
            .ge(0.5)
            .long(),
            "base_phrase_features": batch["base_phrase_features"],
        }
        self.valid_phrase_analysis_metrics[corpus].update(**phrase_analysis_metric_args)
        self.log(
            "valid/word_feature_loss",
            outputs["phrase_analyzer_outputs"]["word_feature_loss"],
        )
        self.log(
            "valid/base_phrase_feature_loss",
            outputs["phrase_analyzer_outputs"]["base_phrase_feature_loss"],
        )

        dependency_parsing_metric_args = {
            "example_ids": batch["example_ids"],
            "dependency_predictions": torch.topk(
                outputs["relation_analyzer_outputs"]["dependency_logits"],
                self.hparams.k,
                dim=2,
            ).indices,
            "dependency_type_predictions": torch.argmax(
                outputs["relation_analyzer_outputs"]["dependency_type_logits"],
                dim=3,
            ),
        }
        self.valid_dependency_parsing_metrics[corpus].update(**dependency_parsing_metric_args)
        self.log(
            "valid/dependency_loss",
            outputs["relation_analyzer_outputs"]["dependency_loss"],
        )

        cohesion_analysis_metric_args = {
            "example_ids": batch["example_ids"],
            "output": outputs["relation_analyzer_outputs"]["cohesion_logits"],
            "dataset": self.trainer.val_dataloaders[dataloader_idx or 0].dataset,
        }
        self.valid_cohesion_analysis_metrics[corpus].update(**cohesion_analysis_metric_args)
        self.log(
            "valid/cohesion_loss",
            outputs["relation_analyzer_outputs"]["cohesion_loss"],
        )

        # TODO: evaluate discourse parsing

    def validation_epoch_end(self, validation_step_outputs) -> None:
        f1_scores: dict[str, float] = defaultdict(float)

        for corpus, metric in self.valid_word_analysis_metrics.items():
            for name, value in metric.compute().items():
                if name == "word_analysis_f1":
                    f1_scores["word_analysis_f1"] += value / len(self.valid_word_analysis_metrics)
                self.log(f"valid_{corpus}/{name}", value)
                metric.reset()
        self.log(
            "valid/word_analysis_f1",
            f1_scores["word_analysis_f1"],
        )

        keys = {
            "macro_word_feature_f1",
            "micro_word_feature_f1",
            "macro_base_phrase_feature_f1",
            "micro_base_phrase_feature_f1",
        }
        for corpus, metric in self.valid_phrase_analysis_metrics.items():
            for name, value in metric.compute().items():
                if name in keys:
                    f1_scores[name] += value / len(self.valid_phrase_analysis_metrics)
                self.log(f"valid_{corpus}/{name}", value)
            metric.reset()
        for key in sorted(keys):
            self.log(f"valid/{key}", f1_scores[key])

        self.log("valid/f1", mean(f1_scores.values()))

        for idx, corpus in enumerate(self.valid_corpora):
            dataset = self.trainer.val_dataloaders[idx].dataset
            metric = self.valid_dependency_parsing_metrics[corpus]
            for name, value in metric.compute(dataset.documents).items():
                self.log(f"valid_{corpus}/{name}", value)
            metric.reset()

        for idx, corpus in enumerate(self.valid_corpora):
            dataset = self.trainer.val_dataloaders[idx].dataset
            metric = self.valid_cohesion_analysis_metrics[corpus]
            for rel, val in metric.compute(dataset).to_dict().items():
                for met, sub_val in val.items():
                    self.log(f"valid_{corpus}/{met}_{rel}", sub_val.f1)
            metric.reset()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> None:
        batch["training"] = False
        outputs: dict[str, torch.Tensor] = self(**batch)
        corpus = self.test_corpora[dataloader_idx or 0]
        word_analysis_metric_args = {
            "pos_preds": torch.argmax(outputs["word_analyzer_outputs"]["pos_logits"], dim=-1),
            "pos_labels": batch["mrph_types"][:, :, 0],
            "subpos_preds": torch.argmax(outputs["word_analyzer_outputs"]["subpos_logits"], dim=-1),
            "subpos_labels": batch["mrph_types"][:, :, 1],
            "conjtype_preds": torch.argmax(outputs["word_analyzer_outputs"]["conjtype_logits"], dim=-1),
            "conjtype_labels": batch["mrph_types"][:, :, 2],
            "conjform_preds": torch.argmax(outputs["word_analyzer_outputs"]["conjform_logits"], dim=-1),
            "conjform_labels": batch["mrph_types"][:, :, 3],
        }
        self.test_word_analysis_metrics[corpus].update(**word_analysis_metric_args)
        self.log(
            "test/word_analysis_loss",
            outputs["word_analyzer_outputs"]["loss"],
        )

        phrase_analysis_metric_args = {
            "example_ids": batch["example_ids"],
            "word_feature_predictions": outputs["phrase_analyzer_outputs"]["word_feature_logits"].ge(0.5).long(),
            "word_features": batch["word_features"],
            "base_phrase_feature_predictions": outputs["phrase_analyzer_outputs"]["base_phrase_feature_logits"]
            .ge(0.5)
            .long(),
            "base_phrase_features": batch["base_phrase_features"],
        }
        self.test_phrase_analysis_metrics[corpus].update(**phrase_analysis_metric_args)
        self.log(
            "test/word_feature_loss",
            outputs["phrase_analyzer_outputs"]["word_feature_loss"],
        )
        self.log(
            "test/base_phrase_feature_loss",
            outputs["phrase_analyzer_outputs"]["base_phrase_feature_loss"],
        )

        dependency_parsing_metric_args = {
            "example_ids": batch["example_ids"],
            "dependency_predictions": torch.topk(
                outputs["relation_analyzer_outputs"]["dependency_logits"],
                self.hparams.k,
                dim=2,
            ).indices,
            "dependency_type_predictions": torch.argmax(
                outputs["relation_analyzer_outputs"]["dependency_type_logits"],
                dim=3,
            ),
        }
        self.test_dependency_parsing_metrics[corpus].update(**dependency_parsing_metric_args)
        self.log(
            "test/dependency_loss",
            outputs["relation_analyzer_outputs"]["dependency_loss"],
        )

        cohesion_analysis_metric_args = {
            "example_ids": batch["example_ids"],
            "output": outputs["relation_analyzer_outputs"]["cohesion_logits"],
            "dataset": self.trainer.test_dataloaders[dataloader_idx or 0].dataset,
        }
        self.test_cohesion_analysis_metrics[corpus].update(**cohesion_analysis_metric_args)
        self.log(
            "test/cohesion_loss",
            outputs["relation_analyzer_outputs"]["cohesion_loss"],
        )

        # TODO: evaluate discourse parsing

    def test_epoch_end(self, test_step_outputs) -> None:
        f1_scores: dict[str, float] = defaultdict(float)

        for corpus, metric in self.test_word_analysis_metrics.items():
            for name, value in metric.compute().items():
                if name == "word_analysis_f1":
                    f1_scores[name] += value / len(self.test_word_analysis_metrics)
                self.log(f"test_{corpus}/{name}", value)
                metric.reset()
        self.log(
            "test/word_analysis_f1",
            f1_scores["word_analysis_f1"],
        )

        keys = {
            "macro_word_feature_f1",
            "micro_word_feature_f1",
            "macro_base_phrase_feature_f1",
            "micro_base_phrase_feature_f1",
        }
        for corpus, metric in self.test_phrase_analysis_metrics.items():
            for name, value in metric.compute().items():
                if name in keys:
                    f1_scores[name] += value / len(self.test_phrase_analysis_metrics)
                self.log(f"test_{corpus}/{name}", value)
            metric.reset()
        for key in sorted(keys):
            self.log(f"test/{key}", f1_scores[key])

        self.log("test/f1", mean(f1_scores.values()))

        for idx, corpus in enumerate(self.test_corpora):
            dataset = self.trainer.test_dataloaders[idx].dataset
            metric = self.test_dependency_parsing_metrics[corpus]
            for name, value in metric.compute(dataset.documents).items():
                self.log(f"test_{corpus}/{name}", value)
            metric.reset()

        for idx, corpus in enumerate(self.test_corpora):
            dataset = self.trainer.test_dataloaders[idx].dataset
            metric = self.test_cohesion_analysis_metrics[corpus]
            for rel, val in metric.compute(dataset).to_dict().items():
                for met, sub_val in val.items():
                    self.log(f"test_{corpus}/{met}_{rel}", sub_val.f1)
            metric.reset()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        batch["training"] = False
        outputs: dict[str, torch.Tensor] = self(**batch)
        return {
            "texts": batch["texts"],
            "word_analysis_pos_logits": outputs["word_analyzer_outputs"]["pos_logits"],
            "word_analysis_subpos_logits": outputs["word_analyzer_outputs"]["subpos_logits"],
            "word_analysis_conjtype_logits": outputs["word_analyzer_outputs"]["conjtype_logits"],
            "word_analysis_conjform_logits": outputs["word_analyzer_outputs"]["conjform_logits"],
            "word_feature_logits": outputs["phrase_analyzer_outputs"]["word_feature_logits"],
            "base_phrase_feature_logits": outputs["phrase_analyzer_outputs"]["base_phrase_feature_logits"],
            "dependency_logits": outputs["relation_analyzer_outputs"]["dependency_logits"],
            "dependency_type_logits": outputs["relation_analyzer_outputs"]["dependency_type_logits"],
            "discourse_parsing_logits": outputs["relation_analyzer_outputs"]["discourse_parsing_logits"],
        }

    def configure_optimizers(self):
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ("bias", "LayerNorm.weight")
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.hparams.optimizer.weight_decay,
                "name": "decay",
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
                "name": "no_decay",
            },
        ]
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer,
            params=optimizer_grouped_parameters,
            _convert_="partial",
        )

        warmup_steps = self.hparams.warmup_steps
        lr_scheduler = hydra.utils.instantiate(
            self.hparams.scheduler,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
