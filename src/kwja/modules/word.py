import os
from functools import reduce
from statistics import mean
from typing import Any, Dict, List

import hydra
import torch
from omegaconf import DictConfig
from transformers import PretrainedConfig, PreTrainedModel

from kwja.modules.base import BaseModule
from kwja.modules.components.crf import CRF
from kwja.modules.components.head import (
    LoRARelationWiseWordSelectionHead,
    LoRASequenceMultiLabelingHead,
    SequenceLabelingHead,
    WordSelectionHead,
)
from kwja.modules.components.pooling import PoolingStrategy, pool_subwords
from kwja.modules.functions.loss import (
    compute_cohesion_analysis_loss,
    compute_multi_label_token_mean_loss,
    compute_token_mean_loss,
    mask_logits,
)
from kwja.utils.constants import (
    BASE_PHRASE_FEATURES,
    CONJFORM_TAGS,
    CONJTYPE_TAGS,
    DEPENDENCY_TYPES,
    DISCOURSE_RELATIONS,
    IGNORE_INDEX,
    NE_TAGS,
    POS_TAGS,
    RESOURCE_PATH,
    SUBPOS_TAGS,
    WORD_FEATURES,
    CohesionTask,
    WordTask,
)
from kwja.utils.reading_prediction import get_reading2reading_id

if os.environ.get("KWJA_CLI_MODE") == "1":
    from kwja.modules.base import DummyModuleMetric as WordModuleMetric  # dummy class for faster loading
else:
    from kwja.metrics import WordModuleMetric  # type: ignore


class WordModule(BaseModule[WordModuleMetric]):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__(hparams, WordModuleMetric(hparams.max_seq_length))

        self.training_tasks: List[WordTask] = list(map(WordTask, self.hparams.training_tasks))

        self.encoder: PreTrainedModel = hydra.utils.call(hparams.encoder.from_config)
        pretrained_model_config: PretrainedConfig = self.encoder.config
        if hasattr(hparams, "special_tokens"):
            self.encoder.resize_token_embeddings(pretrained_model_config.vocab_size + len(hparams.special_tokens))
        head_kwargs: Dict[str, Any] = dict(hidden_size=self.encoder.config.hidden_size, hidden_dropout_prob=0.05)

        # ---------- reading prediction ----------
        self.reading_id2reading: Dict[int, str] = {
            v: k for k, v in get_reading2reading_id(RESOURCE_PATH / "reading_prediction" / "vocab.txt").items()
        }
        self.reading_tagger = SequenceLabelingHead(len(self.reading_id2reading), **head_kwargs)

        # ---------- morphological analysis ----------
        self.pos_tagger = SequenceLabelingHead(len(POS_TAGS), **head_kwargs)
        self.subpos_tagger = SequenceLabelingHead(len(SUBPOS_TAGS), **head_kwargs)
        self.conjtype_tagger = SequenceLabelingHead(len(CONJTYPE_TAGS), **head_kwargs)
        self.conjform_tagger = SequenceLabelingHead(len(CONJFORM_TAGS), **head_kwargs)

        # ---------- word feature tagging ----------
        self.word_feature_tagger = LoRASequenceMultiLabelingHead(len(WORD_FEATURES), **head_kwargs)

        # ---------- named entity recognition ----------
        self.ne_tagger = SequenceLabelingHead(len(NE_TAGS), **head_kwargs)
        self.crf = CRF(NE_TAGS)

        # ---------- base phrase feature tagging ----------
        self.base_phrase_feature_tagger = LoRASequenceMultiLabelingHead(len(BASE_PHRASE_FEATURES), **head_kwargs)

        # ---------- dependency parsing ----------
        self.dependency_topk: int = hparams.dependency_topk
        self.dependency_parser = WordSelectionHead(1, **head_kwargs)
        self.dependency_type_parser = SequenceLabelingHead(
            len(DEPENDENCY_TYPES),
            pretrained_model_config.hidden_size * 2,
            pretrained_model_config.hidden_dropout_prob,
        )

        # ---------- cohesion analysis ----------
        self.cohesion_analyzer = LoRARelationWiseWordSelectionHead(
            self._get_num_cohesion_rels(hparams), rank=1, **head_kwargs
        )

        # ---------- discourse relation analysis ----------
        self.discourse_threshold: float = hparams.discourse_threshold
        self.discourse_relation_analyzer = WordSelectionHead(len(DISCOURSE_RELATIONS), **head_kwargs)

    @staticmethod
    def _get_num_cohesion_rels(hparams: DictConfig) -> int:
        cohesion_tasks = [CohesionTask(ct) for ct in hparams.cohesion_tasks]
        num_cohesion_rels = 0
        if CohesionTask.PAS_ANALYSIS in cohesion_tasks:
            num_cohesion_rels += len(hparams.pas_cases)
        if CohesionTask.BRIDGING_REFERENCE_RESOLUTION in cohesion_tasks:
            num_cohesion_rels += len(hparams.br_cases)
        if CohesionTask.COREFERENCE_RESOLUTION in cohesion_tasks:
            num_cohesion_rels += 1
        return num_cohesion_rels

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.encoder = hydra.utils.call(self.hparams.encoder.from_pretrained)
            if hasattr(self.hparams, "special_tokens"):
                self.encoder.resize_token_embeddings(self.encoder.config.vocab_size + len(self.hparams.special_tokens))

    def forward(self, batch: Any) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            special_token_indices=batch["special_token_indices"],
        )  # (b, seq, hid)
        pooled = pool_subwords(encoded.last_hidden_state, batch["subword_map"], PoolingStrategy.FIRST)  # (b, seq, hid)

        dependency_logits = self.dependency_parser(pooled)  # (b, seq, seq, 1)
        masked_dependency_logits = mask_logits(dependency_logits.squeeze(3), batch["dependency_mask"])
        if batch["dependency_labels"].numel() == 0:
            dependency_labels = masked_dependency_logits.topk(self.dependency_topk, dim=2).indices.unsqueeze(3)
        else:
            dependency_labels_mask = batch["dependency_labels"].ne(IGNORE_INDEX)
            dependency_labels = (batch["dependency_labels"] * dependency_labels_mask).unsqueeze(2).unsqueeze(3)
        unsqueezed = pooled.unsqueeze(2)
        head_hidden_states = torch.take_along_dim(unsqueezed, dependency_labels, dim=1)

        cohesion_logits = self.cohesion_analyzer(pooled)  # (b, seq, seq, rel)
        cohesion_logits = cohesion_logits.permute(0, 3, 1, 2).contiguous()  # -> (b, rel, seq, seq)
        masked_cohesion_logits = mask_logits(cohesion_logits, batch["cohesion_mask"])

        return {
            "reading_logits": self.reading_tagger(encoded.last_hidden_state),
            "pos_logits": self.pos_tagger(pooled),
            "subpos_logits": self.subpos_tagger(pooled),
            "conjtype_logits": self.conjtype_tagger(pooled),
            "conjform_logits": self.conjform_tagger(pooled),
            "word_feature_probabilities": self.word_feature_tagger(pooled),
            "ne_logits": self.ne_tagger(pooled),
            "base_phrase_feature_probabilities": self.base_phrase_feature_tagger(pooled),
            "dependency_logits": masked_dependency_logits,
            "dependency_type_logits": self.dependency_type_parser(
                torch.cat([unsqueezed.expand(head_hidden_states.shape), head_hidden_states], dim=3)
            ),
            "cohesion_logits": masked_cohesion_logits,
            "discourse_logits": self.discourse_relation_analyzer(pooled),
        }

    def training_step(self, batch: Any) -> torch.Tensor:
        ret: Dict[str, torch.Tensor] = self(batch)
        loss_log: Dict[str, torch.Tensor] = {}
        if WordTask.READING_PREDICTION in self.training_tasks:
            loss_log["reading_prediction_loss"] = compute_token_mean_loss(
                ret["reading_logits"], batch["reading_labels"]
            )
        if WordTask.MORPHOLOGICAL_ANALYSIS in self.training_tasks:
            morphological_analysis_losses = [
                compute_token_mean_loss(ret["pos_logits"], batch["pos_labels"]),
                compute_token_mean_loss(ret["subpos_logits"], batch["subpos_labels"]),
                compute_token_mean_loss(ret["conjtype_logits"], batch["conjtype_labels"]),
                compute_token_mean_loss(ret["conjform_logits"], batch["conjform_labels"]),
            ]
            loss_log["morphological_analysis_loss"] = torch.stack(morphological_analysis_losses).mean()
        if WordTask.WORD_FEATURE_TAGGING in self.training_tasks:
            loss_log["word_feature_tagging_loss"] = compute_multi_label_token_mean_loss(
                ret["word_feature_probabilities"],
                batch["word_feature_labels"],
            )
        if WordTask.NER in self.training_tasks:
            loss_log["ner_loss"] = self.crf(ret["ne_logits"], batch["ne_labels"], mask=batch["ne_mask"])
        if WordTask.BASE_PHRASE_FEATURE_TAGGING in self.training_tasks:
            loss_log["base_phrase_feature_tagging_loss"] = compute_multi_label_token_mean_loss(
                ret["base_phrase_feature_probabilities"],
                batch["base_phrase_feature_labels"],
            )
        if WordTask.DEPENDENCY_PARSING in self.training_tasks:
            loss_log["dependency_parsing_loss"] = compute_token_mean_loss(
                ret["dependency_logits"], batch["dependency_labels"]
            )
            top1 = ret["dependency_type_logits"][:, :, 0]
            loss_log["dependency_type_parsing_loss"] = compute_token_mean_loss(top1, batch["dependency_type_labels"])
        if WordTask.COHESION_ANALYSIS in self.training_tasks:
            loss_log["cohesion_analysis_loss"] = compute_cohesion_analysis_loss(
                ret["cohesion_logits"], batch["cohesion_labels"]
            )
        if WordTask.DISCOURSE_RELATION_ANALYSIS in self.training_tasks:
            loss_log["discourse_relation_analysis_loss"] = compute_token_mean_loss(
                ret["discourse_logits"], batch["discourse_labels"]
            )
        self.log_dict({f"train/{key}": value for key, value in loss_log.items()})
        return torch.stack(list(loss_log.values())).sum()

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        kwargs = self.predict_step(batch)
        kwargs.update({"discourse_labels": batch["discourse_labels"]})
        metric = self.valid_corpus2metric[self.valid_corpora[dataloader_idx]]
        metric.update(kwargs)

    def on_validation_epoch_end(self) -> None:
        metrics_log: Dict[str, Dict[str, float]] = {}
        for corpus, metric in self.valid_corpus2metric.items():
            metric.set_properties(
                {
                    "dataset": self.trainer.val_dataloaders[corpus].dataset,
                    "reading_id2reading": self.reading_id2reading,
                    "training_tasks": self.training_tasks,
                }
            )
            metrics = metric.compute()
            if corpus != "kwdlc":
                # discourse labels are available only for KWDLC
                metrics = {
                    key: value for key, value in metrics.items() if not key.startswith("discourse_relation_analysis")
                }
            metrics["aggregated_word_metrics"] = mean(
                metrics[key] for key in self.hparams.aggregating_metrics if key in metrics
            )
            metrics_log[corpus] = metrics
            metric.reset()

        for corpus, metrics in metrics_log.items():
            self.log_dict({f"valid_{corpus}/{key}": value for key, value in metrics.items()})
        for key in reduce(set.union, [set(metrics.keys()) for metrics in metrics_log.values()]):
            mean_score = mean(metrics_log[corpus][key] for corpus in self.valid_corpora if key in metrics_log[corpus])
            self.log(f"valid/{key}", mean_score)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        kwargs = self.predict_step(batch)
        kwargs.update({"discourse_labels": batch["discourse_labels"]})
        metric = self.test_corpus2metric[self.test_corpora[dataloader_idx]]
        metric.update(kwargs)

    def on_test_epoch_end(self) -> None:
        metrics_log: Dict[str, Dict[str, float]] = {}
        for corpus, metric in self.test_corpus2metric.items():
            metric.set_properties(
                {
                    "dataset": self.trainer.test_dataloaders[corpus].dataset,
                    "reading_id2reading": self.reading_id2reading,
                    "training_tasks": self.training_tasks,
                }
            )
            metrics = metric.compute()
            if corpus != "kwdlc":
                # discourse labels are available only for KWDLC
                metrics = {
                    key: value for key, value in metrics.items() if not key.startswith("discourse_relation_analysis")
                }
            metrics["aggregated_word_metrics"] = mean(
                metrics[key] for key in self.hparams.aggregating_metrics if key in metrics
            )
            metrics_log[corpus] = metrics
            metric.reset()

        for corpus, metrics in metrics_log.items():
            self.log_dict({f"test_{corpus}/{key}": value for key, value in metrics.items()})
        for key in reduce(set.union, [set(metrics.keys()) for metrics in metrics_log.values()]):
            mean_score = mean(metrics_log[corpus][key] for corpus in self.test_corpora if key in metrics_log[corpus])
            self.log(f"test/{key}", mean_score)

    def predict_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        ret: Dict[str, torch.Tensor] = self(batch)
        ne_predictions = self.crf.viterbi_decode(ret["ne_logits"], batch["ne_mask"])
        discourse_probabilities = ret["discourse_logits"].softmax(dim=3)
        discourse_max_probabilities, discourse_predictions = discourse_probabilities.max(dim=3)
        discourse_unconfident_indices = discourse_max_probabilities < self.discourse_threshold
        discourse_predictions[discourse_unconfident_indices] = DISCOURSE_RELATIONS.index("談話関係なし")
        return {
            "example_ids": batch["example_ids"],
            "reading_predictions": ret["reading_logits"].argmax(dim=2),
            "reading_subword_map": batch["reading_subword_map"],
            "pos_logits": ret["pos_logits"],
            "subpos_logits": ret["subpos_logits"],
            "conjtype_logits": ret["conjtype_logits"],
            "conjform_logits": ret["conjform_logits"],
            "word_feature_probabilities": ret["word_feature_probabilities"],
            "ne_predictions": ne_predictions,
            "base_phrase_feature_probabilities": ret["base_phrase_feature_probabilities"],
            "dependency_predictions": ret["dependency_logits"].topk(k=self.dependency_topk, dim=2).indices,
            "dependency_type_predictions": ret["dependency_type_logits"].argmax(dim=3),
            "cohesion_logits": ret["cohesion_logits"],
            "discourse_predictions": discourse_predictions,
        }
