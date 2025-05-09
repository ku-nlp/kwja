from collections import defaultdict
from statistics import mean
from typing import Literal, Optional, Union

import torch
from cohesion_tools.evaluators import CohesionEvaluator, CohesionScore
from cohesion_tools.extractors import BridgingExtractor, PasExtractor
from rhoknp import BasePhrase, Document, Morpheme, Phrase, Sentence
from rhoknp.props import DepType, FeatureDict, MemoTag
from seqeval.metrics import f1_score as seqeval_f1_score
from seqeval.scheme import IOB2
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from kwja.callbacks.utils import (  # add_discourse,
    add_base_phrase_features,
    add_cohesion,
    add_dependency,
    add_named_entities,
    build_morphemes,
    chunk_morphemes,
    get_morpheme_attribute_predictions,
)
from kwja.datamodule.datasets import WordDataset
from kwja.metrics.base import BaseModuleMetric
from kwja.metrics.conll18_ud_eval import main as conll18_ud_eval
from kwja.metrics.utils import unique
from kwja.utils.constants import (
    BASE_PHRASE_FEATURES,
    CONJFORM_TAGS,
    CONJTYPE_TAGS,
    DISCOURSE_RELATIONS,
    IGNORE_INDEX,
    IGNORE_VALUE_FEATURE_PAT,
    MASKED,
    POS_TAGS,
    SUBPOS_TAGS,
    WORD_FEATURES,
    CohesionTask,
    WordTask,
)
from kwja.utils.reading_prediction import get_word_level_readings
from kwja.utils.sub_document import extract_target_sentences, to_orig_doc_id


class WordModuleMetric(BaseModuleMetric):
    STATE_NAMES = (
        "example_ids",
        "reading_predictions",
        "reading_subword_map",
        "pos_logits",
        "subpos_logits",
        "conjtype_logits",
        "conjform_logits",
        "word_feature_probabilities",
        "ne_predictions",
        "base_phrase_feature_probabilities",
        "dependency_predictions",
        "dependency_type_predictions",
        "cohesion_logits",
        "discourse_predictions",
        "discourse_labels",
    )

    def __init__(self, max_seq_length: int) -> None:
        super().__init__(max_seq_length)
        self.dataset: Optional[WordDataset] = None
        self.reading_id2reading: Optional[dict[int, str]] = None
        self.training_tasks: Optional[list[WordTask]] = None

        self.example_ids: torch.Tensor
        self.reading_predictions: torch.Tensor
        self.reading_subword_map: torch.Tensor
        self.pos_logits: torch.Tensor
        self.subpos_logits: torch.Tensor
        self.conjtype_logits: torch.Tensor
        self.conjform_logits: torch.Tensor
        self.word_feature_probabilities: torch.Tensor
        self.ne_predictions: torch.Tensor
        self.base_phrase_feature_probabilities: torch.Tensor
        self.dependency_predictions: torch.Tensor
        self.dependency_type_predictions: torch.Tensor
        self.cohesion_logits: torch.Tensor
        self.discourse_predictions: torch.Tensor
        self.discourse_labels: torch.Tensor

    def _pad(self, kwargs: dict[str, torch.Tensor]) -> None:
        for key, value in kwargs.items():
            if key in {"example_ids"} or value.numel() == 0:
                continue
            elif key in {"reading_subword_map", "cohesion_logits", "discourse_predictions", "discourse_labels"}:
                dims = [value.ndim - 2, value.ndim - 1]
            else:
                dims = [1]
            for dim in dims:
                size = [self.max_seq_length - s if i == dim else s for i, s in enumerate(value.size())]
                if key in {"discourse_labels"}:
                    fill_value: Union[float, int] = IGNORE_INDEX
                else:
                    fill_value = MASKED if torch.is_floating_point(value) else 0
                padding = torch.full(size, fill_value, dtype=value.dtype, device=value.device)
                value = torch.cat([value, padding], dim=dim)  # noqa: PLW2901
            kwargs[key] = value

    def compute(self) -> dict[str, float]:
        assert self.training_tasks is not None, "training_tasks isn't set"

        if isinstance(self.example_ids, torch.Tensor) is False:
            self.example_ids = torch.cat(self.example_ids, dim=0)  # type: ignore

        sorted_indices = unique(self.example_ids)
        for state_name in self.STATE_NAMES:
            state = getattr(self, state_name)
            if isinstance(state, torch.Tensor) is False:
                state = torch.cat(state, dim=0)
            if state_name == "reading_subword_map":
                state = state.bool()
            setattr(self, state_name, state[sorted_indices])

        predicted_documents, partly_gold_document1, partly_gold_document2, gold_documents = self._build_documents()

        metrics: dict[str, float] = {}
        if WordTask.READING_PREDICTION in self.training_tasks:
            metrics.update(self.compute_reading_prediction_metrics(predicted_documents, gold_documents))
        if WordTask.MORPHOLOGICAL_ANALYSIS in self.training_tasks:
            metrics.update(self.compute_morphological_analysis_metrics(predicted_documents, gold_documents))
        if WordTask.WORD_FEATURE_TAGGING in self.training_tasks:
            metrics.update(self.compute_word_feature_tagging_metrics(predicted_documents, gold_documents))
        if WordTask.NER in self.training_tasks:
            metrics.update(self.compute_ner_metrics(partly_gold_document1, gold_documents))
        if WordTask.BASE_PHRASE_FEATURE_TAGGING in self.training_tasks:
            metrics.update(self.compute_base_phrase_feature_tagging_metrics(partly_gold_document1, gold_documents))
        if WordTask.DEPENDENCY_PARSING in self.training_tasks:
            metrics.update(self.compute_dependency_parsing_metrics(partly_gold_document1, gold_documents))
        if WordTask.COHESION_ANALYSIS in self.training_tasks:
            metrics.update(self.compute_cohesion_analysis_metrics(partly_gold_document2, gold_documents))
        if WordTask.DISCOURSE_RELATION_ANALYSIS in self.training_tasks:
            predictions = self.discourse_predictions.view(-1)
            labels = self.discourse_labels.view(-1)
            metrics.update(self.compute_discourse_relation_analysis_metrics(predictions, labels))
        return metrics

    def _build_documents(self) -> tuple[list[Document], ...]:
        assert self.dataset is not None, "dataset isn't set"
        assert self.reading_id2reading is not None, "reading_id2reading isn't set"

        doc_id2predicted_sentences: dict[str, list[Sentence]] = defaultdict(list)
        doc_id2partly_gold_sentences1: dict[str, list[Sentence]] = defaultdict(list)
        doc_id2partly_gold_sentences2: dict[str, list[Sentence]] = defaultdict(list)
        doc_id2gold_sentences: dict[str, list[Sentence]] = defaultdict(list)
        for (
            example_id,
            reading_predictions,
            reading_subword_map,
            *morpheme_attribute_logits,  # pos_logits, subpos_logits, conjtype_logits, conjform_logits
            word_feature_probabilities,
            ne_predictions,
            base_phrase_feature_probabilities,
            dependency_predictions,
            dependency_type_predictions,
            cohesion_logits,
            _,  # discourse_predictions
            _,  # discourse_labels
        ) in zip(*[getattr(self, state_name).tolist() for state_name in self.STATE_NAMES]):
            example = self.dataset.examples[example_id]
            gold_document = self.dataset.doc_id2document[example.doc_id]
            orig_doc_id = to_orig_doc_id(gold_document.doc_id)

            word_reading_predictions = get_word_level_readings(
                [self.reading_id2reading[reading_id] for reading_id in reading_predictions],
                [self.dataset.tokenizer.decode(input_id) for input_id in example.encoding.ids],
                reading_subword_map,
            )
            morpheme_attribute_predictions = get_morpheme_attribute_predictions(
                *(logits[: len(gold_document.morphemes)] for logits in morpheme_attribute_logits)
            )
            morphemes = build_morphemes(
                [m.surf for m in gold_document.morphemes],
                [m.lemma for m in gold_document.morphemes],
                word_reading_predictions,
                morpheme_attribute_predictions,
            )
            predicted_document = chunk_morphemes(gold_document, morphemes, word_feature_probabilities)
            predicted_document.doc_id = gold_document.doc_id
            for sentence in extract_target_sentences(predicted_document):
                doc_id2predicted_sentences[orig_doc_id].append(sentence)

            # goldの基本句区切り・基本句主辞を使用
            partly_gold_document1 = gold_document.reparse()
            partly_gold_document1.doc_id = gold_document.doc_id
            self.refresh(partly_gold_document1, level=1)
            for sentence in extract_target_sentences(partly_gold_document1):
                add_named_entities(sentence, ne_predictions)
                add_base_phrase_features(sentence, base_phrase_feature_probabilities)
                add_dependency(
                    sentence, dependency_predictions, dependency_type_predictions, example.special_token_indexer
                )
                doc_id2partly_gold_sentences1[orig_doc_id].append(sentence)

            # goldの基本句区切り・基本句主辞・基本句素性を使用
            partly_gold_document2 = gold_document.reparse()
            partly_gold_document2.doc_id = gold_document.doc_id
            self.refresh(partly_gold_document2, level=2)
            add_cohesion(
                partly_gold_document2,
                cohesion_logits,
                self.dataset.cohesion_task2extractor,
                self.dataset.cohesion_task2rels,
                self.dataset.restrict_cohesion_target,
                example.special_token_indexer,
            )
            for sentence in extract_target_sentences(partly_gold_document2):
                doc_id2partly_gold_sentences2[orig_doc_id].append(sentence)

            for sentence in extract_target_sentences(gold_document):
                doc_id2gold_sentences[orig_doc_id].append(sentence)
        predicted_documents = self._convert_doc_id2sentences_into_documents(doc_id2predicted_sentences)
        partly_gold_documents1 = self._convert_doc_id2sentences_into_documents(doc_id2partly_gold_sentences1)
        partly_gold_documents2 = self._convert_doc_id2sentences_into_documents(doc_id2partly_gold_sentences2)
        gold_documents = self._convert_doc_id2sentences_into_documents(doc_id2gold_sentences)
        return predicted_documents, partly_gold_documents1, partly_gold_documents2, gold_documents

    @staticmethod
    def refresh(document: Document, level: int = 1) -> None:
        """Refresh document

        NOTE:
            level1: clear discourse relations, rel tags, dependencies, and base phrase features.
            level2: clear discourse relations and rel tags.
        """
        assert level in (1, 2), f"invalid level: {level}"
        try:
            for clause in document.clauses:
                clause.discourse_relations.clear()
        except AttributeError:
            pass

        for phrase in document.phrases:
            if level == 1:
                phrase.features.clear()
                phrase.parent_index = None
                phrase.dep_type = None

        for base_phrase in document.base_phrases:
            base_phrase.rel_tags.clear()
            base_phrase.memo_tag = MemoTag()
            if level == 1:
                base_phrase.features.clear()
                base_phrase.parent_index = None
                base_phrase.dep_type = None

    @staticmethod
    def _convert_doc_id2sentences_into_documents(doc_id2sentences: dict[str, list[Sentence]]) -> list[Document]:
        return [Document.from_sentences(ss) for ss in doc_id2sentences.values()]

    @staticmethod
    def compute_reading_prediction_metrics(
        predicted_documents: list[Document], gold_documents: list[Document]
    ) -> dict[str, float]:
        predictions = [m.reading for d in predicted_documents for m in d.morphemes]
        labels = [m.reading for d in gold_documents for m in d.morphemes]
        return {"reading_prediction_accuracy": accuracy_score(y_true=labels, y_pred=predictions)}

    @staticmethod
    def compute_morphological_analysis_metrics(
        predicted_documents: list[Document], gold_documents: list[Document]
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for attribute_name, supported_attribute_values in zip(
            ("pos", "subpos", "conjtype", "conjform"), (POS_TAGS, SUBPOS_TAGS, CONJTYPE_TAGS, CONJFORM_TAGS)
        ):
            metrics[f"{attribute_name}_f1"] = WordModuleMetric._calc_morpheme_attribute_f1_score(
                [m for d in predicted_documents for m in d.morphemes],
                [m for d in gold_documents for m in d.morphemes],
                attribute_name,
                supported_attribute_values,
            )
        metrics["morphological_analysis_f1"] = mean(metrics.values())
        return metrics

    @staticmethod
    def _calc_morpheme_attribute_f1_score(
        predicted_morphemes: list[Morpheme],
        gold_morphemes: list[Morpheme],
        attribute_name: str,
        supported_attribute_values: tuple[str, ...],
    ) -> float:
        labels: list[str] = []
        for morpheme in gold_morphemes:
            attribute_value = getattr(morpheme, attribute_name)
            labels.append(attribute_value if attribute_value in supported_attribute_values else "UNKNOWN")
        predictions: list[str] = [getattr(morpheme, attribute_name) for morpheme in predicted_morphemes]
        return f1_score(y_true=labels, y_pred=predictions, average="micro")

    @staticmethod
    def compute_word_feature_tagging_metrics(
        predicted_documents: list[Document], gold_documents: list[Document]
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        all_labels = []
        all_predictions = []
        for feature_label in WORD_FEATURES:
            labels: list[list[str]] = []
            predictions: list[list[str]] = []
            gold_document: Document
            predicted_document: Document
            for gold_document, predicted_document in zip(gold_documents, predicted_documents):
                if feature_label == "基本句-区切":
                    label = WordModuleMetric._convert_units_into_segmentation_tags(gold_document.base_phrases)
                    prediction = WordModuleMetric._convert_units_into_segmentation_tags(predicted_document.base_phrases)
                elif feature_label == "文節-区切":
                    label = WordModuleMetric._convert_units_into_segmentation_tags(gold_document.phrases)
                    prediction = WordModuleMetric._convert_units_into_segmentation_tags(predicted_document.phrases)
                else:
                    label = [
                        ("B" if feature_label in WordModuleMetric._convert_features_to_labels(m.features) else "O")
                        for m in gold_document.morphemes
                    ]
                    prediction = [
                        ("B" if feature_label in WordModuleMetric._convert_features_to_labels(m.features) else "O")
                        for m in predicted_document.morphemes
                    ]
                labels.append(label)
                predictions.append(prediction)
            metrics[f"{feature_label}_f1"] = seqeval_f1_score(
                y_true=labels, y_pred=predictions, mode="strict", scheme=IOB2, average="micro"
            ).item()
            all_labels += labels
            all_predictions += predictions
        metrics["macro_word_feature_tagging_f1"] = mean(
            metrics[f"{feature_label}_f1"] for feature_label in WORD_FEATURES
        )
        metrics["micro_word_feature_tagging_f1"] = seqeval_f1_score(
            y_true=all_labels, y_pred=all_predictions, mode="strict", scheme=IOB2, average="micro"
        ).item()
        return metrics

    @staticmethod
    def _convert_units_into_segmentation_tags(units: Union[list[Phrase], list[BasePhrase]]) -> list[str]:
        return ["B" if idx == 0 else "I" for u in units for idx in range(len(u.morphemes))]

    @staticmethod
    def compute_ner_metrics(predicted_documents: list[Document], gold_documents: list[Document]) -> dict[str, float]:
        for document in predicted_documents + gold_documents:
            for named_entity in document.named_entities:
                for i, morpheme in enumerate(named_entity.morphemes):
                    bi = "B" if i == 0 else "I"
                    morpheme.features["NE"] = f"{bi}-{named_entity.category.value}"
        labels = [[m.features.get("NE") or "O" for m in d.morphemes] for d in gold_documents]
        predictions = [[m.features.get("NE") or "O" for m in d.morphemes] for d in predicted_documents]
        return {
            "ner_f1": seqeval_f1_score(
                y_true=labels, y_pred=predictions, mode="strict", scheme=IOB2, average="micro"
            ).item()
        }

    @staticmethod
    def compute_base_phrase_feature_tagging_metrics(
        partly_gold_documents1: list[Document], gold_documents: list[Document]
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        all_labels: list[bool] = []
        all_predictions: list[bool] = []
        for feature_label in BASE_PHRASE_FEATURES:
            labels: list[bool] = []
            predictions: list[bool] = []
            gold_document: Document
            predicted_document: Document
            for gold_document, predicted_document in zip(gold_documents, partly_gold_documents1):
                labels += [
                    feature_label in WordModuleMetric._convert_features_to_labels(base_phrase.features)
                    for base_phrase in gold_document.base_phrases
                ]
                predictions += [
                    feature_label in WordModuleMetric._convert_features_to_labels(base_phrase.features)
                    for base_phrase in predicted_document.base_phrases
                ]
            # 正解ラベルがない基本句素性は評価対象外
            if any(labels):
                metrics[f"{feature_label}_f1"] = f1_score(
                    y_true=labels, y_pred=predictions, pos_label=True, average="binary"
                )
            all_labels += labels
            all_predictions += predictions
        metrics["macro_base_phrase_feature_tagging_f1"] = mean(
            metrics[f"{feature_label}_f1"] for feature_label in BASE_PHRASE_FEATURES if f"{feature_label}_f1" in metrics
        )
        metrics["micro_base_phrase_feature_tagging_f1"] = f1_score(
            y_true=all_labels, y_pred=all_predictions, pos_label=True, average="binary"
        )
        return metrics

    @staticmethod
    def _convert_features_to_labels(features: FeatureDict) -> set[str]:
        # {"用言": "動", "体言": True, "節-機能-原因・理由": "ので"} -> {"用言:動", "体言", "節-機能-原因・理由"}
        return {
            k + (f":{v}" if isinstance(v, str) and IGNORE_VALUE_FEATURE_PAT.match(k) is None else "")
            for k, v in features.items()
            if v not in (None, False)
        }

    @staticmethod
    def compute_dependency_parsing_metrics(
        partly_gold_documents1: list[Document], gold_documents: list[Document]
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        unit_names: list[Literal["base_phrase", "morpheme"]] = ["base_phrase", "morpheme"]
        for unit_name in unit_names:
            metrics.update(
                {
                    f"{unit_name}_{k}": v
                    for k, v in conll18_ud_eval(
                        *WordModuleMetric._to_conll_lines(partly_gold_documents1, gold_documents, unit_name)
                    ).items()
                }
            )
        return metrics

    @staticmethod
    def _to_conll_lines(
        partly_gold_documents1: list[Document],
        gold_documents: list[Document],
        unit_name: Literal["base_phrase", "morpheme"],
    ) -> tuple[list[str], list[str]]:
        gold_lines = []
        system_lines = []
        for gold_document, partly_gold_document1 in zip(gold_documents, partly_gold_documents1):
            for sentence in gold_document.sentences:
                gold_lines += [WordModuleMetric._to_conll_line(u) for u in getattr(sentence, f"{unit_name}s")]
                gold_lines.append("\n")
            for sentence in partly_gold_document1.sentences:
                system_lines += [WordModuleMetric._to_conll_line(u) for u in getattr(sentence, f"{unit_name}s")]
                system_lines.append("\n")
        return gold_lines, system_lines

    @staticmethod
    def _to_conll_line(unit: Union[BasePhrase, Morpheme]) -> str:
        id_ = unit.index + 1  # 0-origin -> 1-origin
        if isinstance(unit, BasePhrase):
            form = "".join(m.surf for m in unit.morphemes)
            lemma = "".join(m.lemma for m in unit.morphemes)
            head = unit.parent_index + 1 if unit.parent_index is not None else 0
            deprel = unit.dep_type or DepType.DEPENDENCY
        else:
            assert isinstance(unit, Morpheme)
            form = unit.surf
            lemma = unit.lemma
            head = unit.parent.index + 1 if unit.parent is not None else 0
            if unit == unit.base_phrase.head and unit.base_phrase.dep_type is not None:
                deprel = unit.base_phrase.dep_type
            else:
                deprel = DepType.DEPENDENCY
        upos = xpos = feats = deps = misc = "_"
        return "\t".join(map(str, (id_, form, lemma, upos, xpos, feats, head, deprel, deps, misc)))

    def compute_cohesion_analysis_metrics(
        self, partly_gold_documents2: list[Document], gold_documents: list[Document]
    ) -> dict[str, float]:
        assert self.dataset is not None, "dataset isn't set"
        pas_extractor = self.dataset.cohesion_task2extractor[CohesionTask.PAS_ANALYSIS]
        assert isinstance(pas_extractor, PasExtractor), "pas utils isn't set correctly"
        bridging_extractor = self.dataset.cohesion_task2extractor[CohesionTask.BRIDGING_REFERENCE_RESOLUTION]
        assert isinstance(bridging_extractor, BridgingExtractor), "bridging utils isn't set correctly"

        evaluator = CohesionEvaluator(
            tasks=[task.to_cohesion_tools_task() for task in self.dataset.cohesion_tasks],
            exophora_referent_types=self.dataset.exophora_referent_types,
            pas_cases=pas_extractor.cases,
            bridging_rel_types=bridging_extractor.rel_types,
        )
        evaluator.pas_evaluator.is_target_predicate = lambda p: pas_extractor.is_target(p.base_phrase)
        score: CohesionScore = evaluator.run(partly_gold_documents2, gold_documents)

        metrics: dict[str, float] = {}
        for task_str, analysis_type_to_metric in score.to_dict().items():
            for analysis_type, metric in analysis_type_to_metric.items():
                key = task_str
                if analysis_type != "all":
                    key += f"_{analysis_type}"
                metrics[key + "_f1"] = metric.f1
        metrics["cohesion_analysis_f1"] = mean(
            metrics[key] for key in ("pas_f1", "bridging_f1", "coreference_f1") if key in metrics
        )
        return metrics

    @staticmethod
    def compute_discourse_relation_analysis_metrics(
        predictions: torch.Tensor,  # (N)
        labels: torch.Tensor,  # (N)
    ) -> dict[str, float]:
        mask = labels.ne(IGNORE_INDEX)
        predictions = predictions[mask].cpu()
        labels = labels[mask].cpu()

        if labels.numel() == 0:
            accuracy = 0.0
        else:
            accuracy = accuracy_score(y_true=labels, y_pred=predictions)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true=labels,
            y_pred=predictions,
            labels=[idx for idx, relation in enumerate(DISCOURSE_RELATIONS) if relation != "談話関係なし"],
            average="micro",
            zero_division=0.0,
        )

        return {
            "discourse_relation_analysis_accuracy": accuracy,
            "discourse_relation_analysis_precision": precision,
            "discourse_relation_analysis_recall": recall,
            "discourse_relation_analysis_f1": f1,
        }
