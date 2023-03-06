from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
from omegaconf import DictConfig
from transformers import PretrainedConfig, PreTrainedModel

from kwja.metrics import WordModuleMetric
from kwja.modules.base import BaseModule
from kwja.modules.components.crf import CRF
from kwja.modules.components.head import SequenceLabelingHead, WordSelectionHead
from kwja.modules.components.pooling import PoolingStrategy, pool_subwords
from kwja.modules.functions.loss import compute_multi_label_token_mean_loss, compute_token_mean_loss, mask_logits
from kwja.utils.cohesion_analysis import BridgingUtils, CoreferenceUtils, PasUtils
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


class WordModule(BaseModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__(hparams)

        self.training_tasks: List[WordTask] = list(map(WordTask, self.hparams.training_tasks))

        if valid_corpora := getattr(hparams.datamodule, "valid", None):
            self.valid_corpora: List[str] = list(valid_corpora)
            self.valid_corpus2word_module_metric: Dict[str, WordModuleMetric] = {
                corpus: WordModuleMetric() for corpus in self.valid_corpora
            }
        if test_corpora := getattr(hparams.datamodule, "test", None):
            self.test_corpora: List[str] = list(test_corpora)
            self.test_corpus2word_module_metric: Dict[str, WordModuleMetric] = {
                corpus: WordModuleMetric() for corpus in self.test_corpora
            }

        self.word_encoder: PreTrainedModel = hydra.utils.call(hparams.encoder)
        pretrained_model_config: PretrainedConfig = self.word_encoder.config
        if hasattr(hparams, "special_tokens"):
            self.word_encoder.resize_token_embeddings(pretrained_model_config.vocab_size + len(hparams.special_tokens))
        head_args: Tuple[int, float] = (
            pretrained_model_config.hidden_size,
            pretrained_model_config.hidden_dropout_prob,
        )

        # ---------- reading prediction ----------
        reading2reading_id: Dict[str, int] = get_reading2reading_id(RESOURCE_PATH / "reading_prediction" / "vocab.txt")
        self.reading_id2reading: Dict[int, str] = {v: k for k, v in reading2reading_id.items()}
        self.reading_tagger = SequenceLabelingHead(len(reading2reading_id), *head_args)

        # ---------- morphological analysis ----------
        self.pos_tagger = SequenceLabelingHead(len(POS_TAGS), *head_args)
        self.subpos_tagger = SequenceLabelingHead(len(SUBPOS_TAGS), *head_args)
        self.conjtype_tagger = SequenceLabelingHead(len(CONJTYPE_TAGS), *head_args)
        self.conjform_tagger = SequenceLabelingHead(len(CONJFORM_TAGS), *head_args)

        # ---------- word feature tagging ----------
        self.word_feature_tagger = SequenceLabelingHead(len(WORD_FEATURES), *head_args, multi_label=True)

        # ---------- named entity recognition ----------
        self.ne_tagger = SequenceLabelingHead(len(NE_TAGS), *head_args)
        self.crf = CRF(NE_TAGS)

        # ---------- base phrase feature tagging ----------
        self.base_phrase_feature_tagger = SequenceLabelingHead(len(BASE_PHRASE_FEATURES), *head_args, multi_label=True)

        # ---------- dependency parsing ----------
        self.topk: int = hparams.dependency_topk
        self.dependency_parser = WordSelectionHead(1, *head_args)
        self.dependency_type_parser = SequenceLabelingHead(
            len(DEPENDENCY_TYPES),
            pretrained_model_config.hidden_size * 2,
            pretrained_model_config.hidden_dropout_prob,
        )

        # ---------- cohesion analysis ----------
        self.cohesion_analyzer = WordSelectionHead(self._get_num_cohesion_rels(hparams), *head_args)

        # ---------- discourse parsing ----------
        self.discourse_parsing_threshold: float = hparams.discourse_parsing_threshold
        self.discourse_parser = WordSelectionHead(len(DISCOURSE_RELATIONS), *head_args)

    @staticmethod
    def _get_num_cohesion_rels(hparams: DictConfig) -> int:
        num_cohesion_rels = 0
        kwargs = {
            "exophora_referents": hparams.exophora_referents,
            "restrict_target": hparams.restrict_cohesion_target,
        }
        for cohesion_task in [CohesionTask(t) for t in hparams.cohesion_tasks]:
            if cohesion_task == CohesionTask.PAS_ANALYSIS:
                pas_utils = PasUtils(hparams.pas_cases, "all", **kwargs)
                num_cohesion_rels += len(pas_utils.rels)
            elif cohesion_task == CohesionTask.BRIDGING_REFERENCE_RESOLUTION:
                bridging_utils = BridgingUtils(hparams.br_cases, **kwargs)
                num_cohesion_rels += len(bridging_utils.rels)
            elif cohesion_task == CohesionTask.COREFERENCE_RESOLUTION:
                coreference_utils = CoreferenceUtils(**kwargs)
                num_cohesion_rels += len(coreference_utils.rels)
        return num_cohesion_rels

    def forward(self, batch: Any) -> Dict[str, torch.Tensor]:
        # (b, seq, hid)
        encoded = self.word_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        pooled = pool_subwords(encoded.last_hidden_state, batch["subword_map"], PoolingStrategy.FIRST)

        dependency_logits = self.dependency_parser(pooled)
        masked_dependency_logits = mask_logits(dependency_logits.squeeze(3), batch["dependency_mask"])
        if 0 not in batch["dependency_labels"].size():
            dependency_labels = batch["dependency_labels"]
            dependency_labels = (dependency_labels * dependency_labels.ne(IGNORE_INDEX)).unsqueeze(2).unsqueeze(3)
            topk = 1
        else:
            dependency_labels = masked_dependency_logits.topk(self.topk, dim=2).indices.unsqueeze(3)
            topk = self.topk
        unsqueezed = pooled.unsqueeze(2)
        batch_size, seq_len, hidden_size = pooled.shape
        source_shape = (batch_size, seq_len, seq_len, hidden_size)
        target_shape = (batch_size, seq_len, topk, hidden_size)
        # gather は IGNORE_INDEX を渡せない
        head_hidden_states = unsqueezed.expand(source_shape).gather(2, dependency_labels.expand(target_shape))

        cohesion_logits = self.cohesion_analyzer(pooled)
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
                torch.cat([unsqueezed.expand(target_shape), head_hidden_states], dim=3)
            ),
            "cohesion_logits": masked_cohesion_logits,
            "discourse_logits": self.discourse_parser(pooled),
        }

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        ret: Dict[str, torch.Tensor] = self(batch)
        loss = torch.tensor(0.0, device=self.device)

        if WordTask.READING_PREDICTION in self.training_tasks:
            reading_prediction_loss = compute_token_mean_loss(ret["reading_logits"], batch["reading_labels"])
            loss += reading_prediction_loss
            self.log("train/reading_prediction_loss", reading_prediction_loss)

        if WordTask.MORPHOLOGICAL_ANALYSIS in self.training_tasks:
            pos_loss = compute_token_mean_loss(ret["pos_logits"], batch["pos_labels"])
            subpos_loss = compute_token_mean_loss(ret["subpos_logits"], batch["subpos_labels"])
            conjtype_loss = compute_token_mean_loss(ret["conjtype_logits"], batch["conjtype_labels"])
            conjform_loss = compute_token_mean_loss(ret["conjform_logits"], batch["conjform_labels"])
            morphological_analysis_loss = (pos_loss + subpos_loss + conjtype_loss + conjform_loss) / 4
            loss += morphological_analysis_loss
            self.log("train/morphological_analysis_loss", morphological_analysis_loss)

        if WordTask.WORD_FEATURE_TAGGING in self.training_tasks:
            word_feature_mask = batch["word_feature_labels"].ne(IGNORE_INDEX)
            word_feature_tagging_loss = compute_multi_label_token_mean_loss(
                ret["word_feature_probabilities"] * word_feature_mask,
                batch["word_feature_labels"].float() * word_feature_mask,
                word_feature_mask,
            )
            loss += word_feature_tagging_loss
            self.log("train/word_feature_tagging_loss", word_feature_tagging_loss)

        if WordTask.NER in self.training_tasks:
            ne_labels = torch.where(batch["target_mask"] == 1, batch["ne_labels"], NE_TAGS.index("O"))
            ner_loss = self.crf(ret["ne_logits"], ne_labels, mask=batch["target_mask"])
            loss += ner_loss
            self.log("train/ner_loss", ner_loss)

        if WordTask.BASE_PHRASE_FEATURE_TAGGING in self.training_tasks:
            base_phrase_feature_mask = batch["base_phrase_feature_labels"].ne(IGNORE_INDEX)
            base_phrase_feature_tagging_loss = compute_multi_label_token_mean_loss(
                ret["base_phrase_feature_probabilities"] * base_phrase_feature_mask,
                batch["base_phrase_feature_labels"].float() * base_phrase_feature_mask,
                base_phrase_feature_mask,
            )
            loss += base_phrase_feature_tagging_loss
            self.log("train/base_phrase_feature_tagging_loss", base_phrase_feature_tagging_loss)

        if WordTask.DEPENDENCY_PARSING in self.training_tasks:
            dependency_parsing_loss = compute_token_mean_loss(ret["dependency_logits"], batch["dependency_labels"])
            top1 = ret["dependency_type_logits"][:, :, 0, :]
            dependency_type_parsing_loss = compute_token_mean_loss(top1, batch["dependency_type_labels"])
            loss += dependency_parsing_loss
            loss += dependency_type_parsing_loss
            self.log("train/dependency_parsing_loss", dependency_parsing_loss)
            self.log("train/dependency_type_parsing_loss", dependency_type_parsing_loss)

        if WordTask.COHESION_ANALYSIS in self.training_tasks:
            log_softmax = torch.log_softmax(ret["cohesion_logits"], dim=3)  # (b, rel, seq, seq)
            denominator = batch["cohesion_labels"].sum() + 1e-6
            cohesion_analysis_loss = (-log_softmax * batch["cohesion_labels"]).sum().div(denominator)
            loss += cohesion_analysis_loss
            self.log("train/cohesion_analysis_loss", cohesion_analysis_loss)

        if WordTask.DISCOURSE_PARSING in self.training_tasks:
            discourse_parsing_loss = compute_token_mean_loss(ret["discourse_logits"], batch["discourse_labels"])
            loss += discourse_parsing_loss
            self.log("train/discourse_parsing_loss", discourse_parsing_loss)

        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> None:
        kwargs = self.predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        kwargs.update({"discourse_labels": batch["discourse_labels"]})
        corpus = self.valid_corpora[dataloader_idx or 0]
        self.valid_corpus2word_module_metric[corpus].update(kwargs)

    def validation_epoch_end(self, validation_step_outputs) -> None:
        metrics_log: Dict[str, Dict[str, float]] = {corpus: {} for corpus in self.valid_corpora}
        for corpus, word_module_metric in self.valid_corpus2word_module_metric.items():
            dataset = self.trainer.val_dataloaders[self.valid_corpora.index(corpus)].dataset
            word_module_metric.set_properties(
                {
                    "dataset": dataset,
                    "reading_id2reading": self.reading_id2reading,
                    "training_tasks": self.training_tasks,
                }
            )
            metrics = word_module_metric.compute()
            metrics["aggregated_word_metrics"] = mean(
                metrics[key] for key in self.hparams.aggregating_metrics if key in metrics
            )
            metrics_log[corpus] = metrics
            word_module_metric.reset()

        for corpus, metrics in metrics_log.items():
            self.log_dict({f"valid_{corpus}/{key}": value for key, value in metrics.items()})
        for key in list(metrics_log.values())[0].keys():
            mean_score = mean(metrics_log[corpus][key] for corpus in self.valid_corpora if key in metrics_log[corpus])
            self.log(f"valid/{key}", mean_score)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> None:
        kwargs = self.predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        kwargs.update({"discourse_labels": batch["discourse_labels"]})
        corpus = self.test_corpora[dataloader_idx or 0]
        self.test_corpus2word_module_metric[corpus].update(kwargs)

    def test_epoch_end(self, test_step_outputs) -> None:
        metrics_log: Dict[str, Dict[str, float]] = {corpus: {} for corpus in self.test_corpora}
        for corpus, word_module_metric in self.test_corpus2word_module_metric.items():
            dataset = self.trainer.test_dataloaders[self.test_corpora.index(corpus)].dataset
            word_module_metric.set_properties(
                {
                    "dataset": dataset,
                    "reading_id2reading": self.reading_id2reading,
                    "training_tasks": self.training_tasks,
                }
            )
            metrics = word_module_metric.compute()
            metrics["aggregated_word_metrics"] = mean(
                metrics[key] for key in self.hparams.aggregating_metrics if key in metrics
            )
            metrics_log[corpus] = metrics
            word_module_metric.reset()

        for corpus, metrics in metrics_log.items():
            self.log_dict({f"test_{corpus}/{key}": value for key, value in metrics.items()})
        for key in list(metrics_log.values())[0].keys():
            mean_score = mean(metrics_log[corpus][key] for corpus in self.test_corpora if key in metrics_log[corpus])
            self.log(f"test/{key}", mean_score)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        ret: Dict[str, torch.Tensor] = self(batch)
        ne_predictions = self.crf.viterbi_decode(ret["ne_logits"], batch["target_mask"])
        discourse_probabilities = ret["discourse_logits"].softmax(dim=3)
        discourse_max_probabilities, discourse_predictions = discourse_probabilities.max(dim=3)
        discourse_unconfident_indices = discourse_max_probabilities < self.discourse_parsing_threshold
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
            "dependency_predictions": ret["dependency_logits"].topk(k=self.topk, dim=2).indices,
            "dependency_type_predictions": ret["dependency_type_logits"].argmax(dim=3),
            "cohesion_logits": ret["cohesion_logits"],
            "discourse_predictions": discourse_predictions,
        }
