from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.core.lightning import LightningModule
from transformers import PretrainedConfig

from kwja.evaluators.cohesion_analysis_metric import CohesionAnalysisMetric
from kwja.evaluators.dependency_parsing_metric import DependencyParsingMetric
from kwja.evaluators.discourse_parsing_metric import DiscourseParsingMetric
from kwja.evaluators.ner_metric import NERMetric
from kwja.evaluators.phrase_analysis_metric import PhraseAnalysisMetric
from kwja.evaluators.reading_predictor_metric import ReadingPredictorMetric
from kwja.evaluators.word_analysis_metric import WordAnalysisMetric
from kwja.models.models.phrase_analyzer import PhraseAnalyzer
from kwja.models.models.pooling import PoolingStrategy
from kwja.models.models.reading_predictor import ReadingPredictor
from kwja.models.models.relation_analyzer import RelationAnalyzer
from kwja.models.models.word_analyzer import WordAnalyzer
from kwja.models.models.word_encoder import WordEncoder
from kwja.utils.constants import DISCOURSE_RELATIONS
from kwja.utils.util import filter_dict_items


class WordTask(Enum):
    READING_PREDICTION = "reading_prediction"
    WORD_ANALYSIS = "word_analysis"
    NER = "ner"
    WORD_FEATURE_TAGGING = "word_feature_tagging"
    BASE_PHRASE_FEATURE_TAGGING = "base_phrase_feature_tagging"
    DEPENDENCY_PARSING = "dependency_parsing"
    COHESION_ANALYSIS = "cohesion_analysis"
    DISCOURSE_PARSING = "discourse_parsing"


class WordModule(LightningModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__()
        OmegaConf.resolve(hparams)
        self.save_hyperparameters(hparams)
        self.training_tasks = list(map(WordTask, self.hparams.training_tasks))

        self.valid_corpora = list(hparams.datamodule.valid.keys()) if "valid" in hparams.datamodule else []
        self.test_corpora = list(hparams.datamodule.test.keys()) if "test" in hparams.datamodule else []

        self.word_encoder: WordEncoder = WordEncoder(hparams)

        pretrained_model_config: PretrainedConfig = self.word_encoder.pretrained_model.config

        self.reading_predictor: ReadingPredictor = ReadingPredictor(
            hparams.dataset.reading_resource_path, pretrained_model_config
        )
        self.valid_reading_predictor_metrics: dict[str, ReadingPredictorMetric] = {
            corpus: ReadingPredictorMetric() for corpus in self.valid_corpora
        }
        self.test_reading_predictor_metrics: dict[str, ReadingPredictorMetric] = {
            corpus: ReadingPredictorMetric() for corpus in self.test_corpora
        }

        self.word_analyzer: WordAnalyzer = WordAnalyzer(pretrained_model_config)
        self.valid_word_analysis_metrics: dict[str, WordAnalysisMetric] = {
            corpus: WordAnalysisMetric() for corpus in self.valid_corpora
        }
        self.test_word_analysis_metrics: dict[str, WordAnalysisMetric] = {
            corpus: WordAnalysisMetric() for corpus in self.test_corpora
        }
        self.valid_ner_metrics: dict[str, NERMetric] = {corpus: NERMetric() for corpus in self.valid_corpora}
        self.test_ner_metrics: dict[str, NERMetric] = {corpus: NERMetric() for corpus in self.test_corpora}

        self.phrase_analyzer: PhraseAnalyzer = PhraseAnalyzer(pretrained_model_config)
        self.valid_phrase_analysis_metrics: dict[str, PhraseAnalysisMetric] = {
            corpus: PhraseAnalysisMetric() for corpus in self.valid_corpora
        }
        self.test_phrase_analysis_metrics: dict[str, PhraseAnalysisMetric] = {
            corpus: PhraseAnalysisMetric() for corpus in self.test_corpora
        }

        self.relation_analyzer: RelationAnalyzer = RelationAnalyzer(hparams, pretrained_model_config)
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
        self.valid_discourse_parsing_metrics: dict[str, DiscourseParsingMetric] = {
            corpus: DiscourseParsingMetric() for corpus in self.valid_corpora
        }
        self.test_discourse_parsing_metrics: dict[str, DiscourseParsingMetric] = {
            corpus: DiscourseParsingMetric() for corpus in self.test_corpora
        }

        self.discourse_parsing_threshold: float = self.hparams.discourse_parsing_threshold

    def forward(self, **batch) -> dict[str, dict[str, torch.Tensor]]:
        # (batch_size, seq_len, hidden_size)
        hast_hidden_states, pooled_outputs = self.word_encoder(batch, PoolingStrategy.FIRST)
        reading_predictor_outputs = self.reading_predictor(hast_hidden_states, batch)
        word_analyzer_outputs = self.word_analyzer(pooled_outputs, batch)
        phrase_analyzer_outputs = self.phrase_analyzer(pooled_outputs, batch)
        relation_analyzer_output = self.relation_analyzer(pooled_outputs, batch)
        return {
            "reading_predictor_outputs": reading_predictor_outputs,
            "word_analyzer_outputs": word_analyzer_outputs,
            "phrase_analyzer_outputs": phrase_analyzer_outputs,
            "relation_analyzer_outputs": relation_analyzer_output,
        }

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        batch["training"] = True
        outputs: dict[str, dict[str, torch.Tensor]] = self(**batch)
        loss = torch.tensor(0.0, device=self.device)
        if WordTask.READING_PREDICTION in self.training_tasks:
            reading_predictor_loss = outputs["reading_predictor_outputs"]["loss"]
            loss += reading_predictor_loss
            self.log("train/reading_predictor_loss", reading_predictor_loss)
        if WordTask.WORD_ANALYSIS in self.training_tasks:
            word_analysis_loss = outputs["word_analyzer_outputs"]["loss"]
            loss += word_analysis_loss
            self.log("train/word_analysis_loss", word_analysis_loss)
        if WordTask.NER in self.training_tasks:
            ner_loss = outputs["phrase_analyzer_outputs"]["ner_loss"]
            loss += ner_loss
            self.log("train/ner_loss", ner_loss)
        if WordTask.WORD_FEATURE_TAGGING in self.training_tasks:
            word_feature_loss = outputs["phrase_analyzer_outputs"]["word_feature_loss"]
            loss += word_feature_loss
            self.log("train/word_feature_loss", word_feature_loss)
        if WordTask.BASE_PHRASE_FEATURE_TAGGING in self.training_tasks:
            base_phrase_feature_loss = outputs["phrase_analyzer_outputs"]["base_phrase_feature_loss"]
            loss += base_phrase_feature_loss
            self.log("train/base_phrase_feature_loss", base_phrase_feature_loss)
        if WordTask.DEPENDENCY_PARSING in self.training_tasks:
            dependency_loss = outputs["relation_analyzer_outputs"]["dependency_loss"]
            loss += dependency_loss
            self.log("train/dependency_loss", dependency_loss)
            dependency_type_loss = outputs["relation_analyzer_outputs"]["dependency_type_loss"]
            loss += dependency_type_loss
            self.log("train/dependency_type_loss", dependency_type_loss)
        if WordTask.COHESION_ANALYSIS in self.training_tasks:
            cohesion_analysis_loss = outputs["relation_analyzer_outputs"]["cohesion_loss"]
            loss += cohesion_analysis_loss
            self.log("train/cohesion_analysis_loss", cohesion_analysis_loss)
        if WordTask.DISCOURSE_PARSING in self.training_tasks:
            discourse_parsing_loss = outputs["relation_analyzer_outputs"]["discourse_parsing_loss"]
            loss += discourse_parsing_loss
            self.log("train/discourse_parsing_loss", discourse_parsing_loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> None:
        batch["training"] = False
        outputs: dict[str, dict[str, torch.Tensor]] = self(**batch)
        corpus = self.valid_corpora[dataloader_idx or 0]

        reading_predictor_metric_args = {
            "predictions": torch.argmax(outputs["reading_predictor_outputs"]["logits"], dim=-1),
            "labels": batch["reading_ids"],
        }
        self.valid_reading_predictor_metrics[corpus].update(**reading_predictor_metric_args)
        self.log("valid/reading_predictor_loss", outputs["reading_predictor_outputs"]["loss"])

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
        self.log("valid/word_analysis_loss", outputs["word_analyzer_outputs"]["loss"])

        ner_metric_args = {
            "example_ids": batch["example_ids"],
            "ne_tag_predictions": torch.argmax(outputs["phrase_analyzer_outputs"]["ne_logits"], dim=-1),
            "ne_tags": batch["ne_tags"],
        }
        self.valid_ner_metrics[corpus].update(**ner_metric_args)
        self.log("valid/ner_loss", outputs["phrase_analyzer_outputs"]["ner_loss"])

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
        self.log("valid/word_feature_loss", outputs["phrase_analyzer_outputs"]["word_feature_loss"])
        self.log("valid/base_phrase_feature_loss", outputs["phrase_analyzer_outputs"]["base_phrase_feature_loss"])

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
        self.log("valid/dependency_loss", outputs["relation_analyzer_outputs"]["dependency_loss"])

        cohesion_analysis_metric_args = {
            "example_ids": batch["example_ids"],
            "output": outputs["relation_analyzer_outputs"]["cohesion_logits"],
            "dataset": self.trainer.val_dataloaders[dataloader_idx or 0].dataset,
        }
        self.valid_cohesion_analysis_metrics[corpus].update(**cohesion_analysis_metric_args)
        self.log("valid/cohesion_loss", outputs["relation_analyzer_outputs"]["cohesion_loss"])

        discourse_parsing_logits = outputs["relation_analyzer_outputs"]["discourse_parsing_logits"]
        discourse_parsing_probs = torch.softmax(discourse_parsing_logits, dim=-1)
        discourse_parsing_max_probs, discourse_parsing_predictions = torch.max(discourse_parsing_probs, dim=-1)
        discourse_parsing_unconfident_indexes = discourse_parsing_max_probs < self.discourse_parsing_threshold
        discourse_parsing_predictions[discourse_parsing_unconfident_indexes] = DISCOURSE_RELATIONS.index("談話関係なし")
        discourse_parsing_metric_args = {
            "discourse_parsing_predictions": discourse_parsing_predictions,
            "discourse_parsing_labels": batch["discourse_relations"],
        }
        self.valid_discourse_parsing_metrics[corpus].update(**discourse_parsing_metric_args)
        self.log("valid/discourse_parsing_loss", outputs["relation_analyzer_outputs"]["discourse_parsing_loss"])

    def validation_epoch_end(self, validation_step_outputs) -> None:
        log_metrics: dict[str, dict[str, float]] = {corpus: {} for corpus in self.valid_corpora}

        for corpus, metric in self.valid_reading_predictor_metrics.items():
            if WordTask.READING_PREDICTION in self.training_tasks:
                log_metrics[corpus].update(metric.compute())
            metric.reset()

        for corpus, metric in self.valid_word_analysis_metrics.items():
            if WordTask.WORD_ANALYSIS in self.training_tasks:
                log_metrics[corpus].update(metric.compute())
            metric.reset()

        for corpus, metric in self.valid_ner_metrics.items():
            if WordTask.NER in self.training_tasks:
                log_metrics[corpus].update(metric.compute())
            metric.reset()

        for corpus, metric in self.valid_phrase_analysis_metrics.items():
            if (
                WordTask.WORD_FEATURE_TAGGING in self.training_tasks
                or WordTask.BASE_PHRASE_FEATURE_TAGGING in self.training_tasks
            ):
                log_metrics[corpus].update(metric.compute())
            metric.reset()

        for idx, corpus in enumerate(self.valid_corpora):
            metric = self.valid_dependency_parsing_metrics[corpus]
            if WordTask.DEPENDENCY_PARSING in self.training_tasks:
                dataset = self.trainer.val_dataloaders[idx].dataset
                documents = [dataset.doc_id2document[example.doc_id] for example in dataset.examples]
                log_metrics[corpus].update(metric.compute(documents))
            metric.reset()

        for idx, corpus in enumerate(self.valid_corpora):
            metric = self.valid_cohesion_analysis_metrics[corpus]
            if WordTask.COHESION_ANALYSIS in self.training_tasks:
                dataset = self.trainer.val_dataloaders[idx].dataset
                _, metric_values = metric.compute(dataset)
                log_metrics[corpus].update(metric_values)
            metric.reset()

        for idx, corpus in enumerate(self.valid_corpora):
            metric = self.valid_discourse_parsing_metrics[corpus]
            if WordTask.DISCOURSE_PARSING in self.training_tasks:
                for name, value in metric.compute().items():
                    log_metrics[corpus][name] = value
            metric.reset()

        for corpus, metrics in log_metrics.items():
            metrics["aggregated_word_metrics"] = mean(
                metrics[key] for key in self.hparams.aggregating_metrics if key in metrics
            )

        for corpus, metrics in log_metrics.items():
            self.log_dict({f"valid_{corpus}/{name}": value for name, value in metrics.items()})
        for name in list(log_metrics.values())[0].keys():
            self.log(f"valid/{name}", mean(log_metrics[corpus].get(name, 0) for corpus in self.valid_corpora))
        self.log(
            "valid/aggregated_word_metrics",
            mean(metrics["aggregated_word_metrics"] for metrics in log_metrics.values()),
        )

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> None:
        batch["training"] = False
        outputs: dict[str, dict[str, torch.Tensor]] = self(**batch)
        corpus = self.test_corpora[dataloader_idx or 0]

        reading_predictor_metric_args = {
            "predictions": torch.argmax(outputs["reading_predictor_outputs"]["logits"], dim=-1),
            "labels": batch["reading_ids"],
        }
        self.test_reading_predictor_metrics[corpus].update(**reading_predictor_metric_args)
        self.log("test/reading_predictor_loss", outputs["reading_predictor_outputs"]["loss"])

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
        self.log("test/word_analysis_loss", outputs["word_analyzer_outputs"]["loss"])

        ner_metric_args = {
            "example_ids": batch["example_ids"],
            "ne_tag_predictions": torch.argmax(outputs["phrase_analyzer_outputs"]["ne_logits"], dim=-1),
            "ne_tags": batch["ne_tags"],
        }
        self.test_ner_metrics[corpus].update(**ner_metric_args)
        self.log("test/ner_loss", outputs["phrase_analyzer_outputs"]["ner_loss"])

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
        self.log("test/word_feature_loss", outputs["phrase_analyzer_outputs"]["word_feature_loss"])
        self.log("test/base_phrase_feature_loss", outputs["phrase_analyzer_outputs"]["base_phrase_feature_loss"])

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
        self.log("test/dependency_loss", outputs["relation_analyzer_outputs"]["dependency_loss"])

        cohesion_analysis_metric_args = {
            "example_ids": batch["example_ids"],
            "output": outputs["relation_analyzer_outputs"]["cohesion_logits"],
            "dataset": self.trainer.test_dataloaders[dataloader_idx or 0].dataset,
        }
        self.test_cohesion_analysis_metrics[corpus].update(**cohesion_analysis_metric_args)
        self.log("test/cohesion_loss", outputs["relation_analyzer_outputs"]["cohesion_loss"])

        discourse_parsing_logits = outputs["relation_analyzer_outputs"]["discourse_parsing_logits"]
        discourse_parsing_probs = torch.softmax(discourse_parsing_logits, dim=-1)
        discourse_parsing_max_probs, discourse_parsing_predictions = torch.max(discourse_parsing_probs, dim=-1)
        discourse_parsing_unconfident_indexes = discourse_parsing_max_probs < self.discourse_parsing_threshold
        discourse_parsing_predictions[discourse_parsing_unconfident_indexes] = DISCOURSE_RELATIONS.index("談話関係なし")
        discourse_parsing_metric_args = {
            "discourse_parsing_predictions": discourse_parsing_predictions,
            "discourse_parsing_labels": batch["discourse_relations"],
        }
        self.test_discourse_parsing_metrics[corpus].update(**discourse_parsing_metric_args)
        self.log("test/discourse_parsing_loss", outputs["relation_analyzer_outputs"]["discourse_parsing_loss"])

    def test_epoch_end(self, test_step_outputs) -> None:
        log_metrics: dict[str, dict[str, float]] = {corpus: {} for corpus in self.test_corpora}

        for corpus, metric in self.test_reading_predictor_metrics.items():
            if WordTask.READING_PREDICTION in self.training_tasks:
                log_metrics[corpus].update(metric.compute())
            metric.reset()

        for corpus, metric in self.test_word_analysis_metrics.items():
            if WordTask.WORD_ANALYSIS in self.training_tasks:
                log_metrics[corpus].update(metric.compute())
            metric.reset()

        for corpus, metric in self.test_ner_metrics.items():
            if WordTask.NER in self.training_tasks:
                log_metrics[corpus].update(metric.compute())
            metric.reset()

        for corpus, metric in self.test_phrase_analysis_metrics.items():
            if (
                WordTask.WORD_FEATURE_TAGGING in self.training_tasks
                or WordTask.BASE_PHRASE_FEATURE_TAGGING in self.training_tasks
            ):
                log_metrics[corpus].update(metric.compute())
            metric.reset()

        for idx, corpus in enumerate(self.test_corpora):
            metric = self.test_dependency_parsing_metrics[corpus]
            if WordTask.DEPENDENCY_PARSING in self.training_tasks:
                dataset = self.trainer.test_dataloaders[idx].dataset
                documents = [dataset.doc_id2document[example.doc_id] for example in dataset.examples]
                log_metrics[corpus].update(metric.compute(documents))
            metric.reset()

        for idx, corpus in enumerate(self.test_corpora):
            metric = self.test_cohesion_analysis_metrics[corpus]
            if WordTask.COHESION_ANALYSIS in self.training_tasks:
                dataset = self.trainer.test_dataloaders[idx].dataset
                score_result, metric_values = metric.compute(dataset)
                log_metrics[corpus].update(metric_values)
                Path(self.hparams.run_dir).mkdir(exist_ok=True)
                score_result.export_csv(f"{self.hparams.run_dir}/cohesion_analysis_scores_{corpus}.csv")
                score_result.export_txt(f"{self.hparams.run_dir}/cohesion_analysis_scores_{corpus}.txt")
            metric.reset()

        for idx, corpus in enumerate(self.test_corpora):
            metric = self.test_discourse_parsing_metrics[corpus]
            if WordTask.DISCOURSE_PARSING in self.training_tasks:
                for name, value in metric.compute().items():
                    log_metrics[corpus][name] = value
            metric.reset()

        for corpus, metrics in log_metrics.items():
            metrics["aggregated_word_metrics"] = mean(
                metrics[key] for key in self.hparams.aggregating_metrics if key in metrics
            )

        for corpus, metrics in log_metrics.items():
            self.log_dict({f"test_{corpus}/{name}": value for name, value in metrics.items()})
        for name in list(log_metrics.values())[0].keys():
            self.log(f"test/{name}", mean(log_metrics[corpus].get(name, 0) for corpus in self.test_corpora))
        self.log(
            "test/aggregated_word_metrics",
            mean(metrics["aggregated_word_metrics"] for metrics in log_metrics.values()),
        )

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        batch["training"] = False
        outputs: dict[str, dict[str, torch.Tensor]] = self(**batch)
        return {
            "tokens": batch["tokens"],
            "example_ids": batch["example_ids"],
            "dataloader_idx": dataloader_idx or 0,
            "reading_subword_map": batch["reading_subword_map"],
            "reading_prediction_logits": outputs["reading_predictor_outputs"]["logits"],
            "word_analysis_pos_logits": outputs["word_analyzer_outputs"]["pos_logits"],
            "word_analysis_subpos_logits": outputs["word_analyzer_outputs"]["subpos_logits"],
            "word_analysis_conjtype_logits": outputs["word_analyzer_outputs"]["conjtype_logits"],
            "word_analysis_conjform_logits": outputs["word_analyzer_outputs"]["conjform_logits"],
            "ne_logits": outputs["phrase_analyzer_outputs"]["ne_logits"],
            "word_feature_logits": outputs["phrase_analyzer_outputs"]["word_feature_logits"],
            "base_phrase_feature_logits": outputs["phrase_analyzer_outputs"]["base_phrase_feature_logits"],
            "dependency_logits": outputs["relation_analyzer_outputs"]["dependency_logits"],
            "dependency_type_logits": outputs["relation_analyzer_outputs"]["dependency_type_logits"],
            "cohesion_logits": outputs["relation_analyzer_outputs"]["cohesion_logits"],
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
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step", "frequency": 1},
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        hparams = checkpoint["hyper_parameters"]["hparams"]
        OmegaConf.set_struct(hparams, False)
        hparams = filter_dict_items(hparams, self.hparams.hparams_to_ignore_on_save)
        checkpoint["hyper_parameters"] = {"hparams": hparams}
