from collections import defaultdict
from itertools import chain
from statistics import mean
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
from cohesion_tools.evaluation import CohesionScore, CohesionScorer
from cohesion_tools.extractors import PasExtractor
from rhoknp import BasePhrase, Document, Morpheme, Phrase, Sentence
from rhoknp.props import DepType, MemoTag
from seqeval.metrics import accuracy_score, f1_score
from seqeval.scheme import IOB2

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
        self.reading_id2reading: Optional[Dict[int, str]] = None
        self.training_tasks: Optional[List[WordTask]] = None

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

    def _pad(self, kwargs: Dict[str, torch.Tensor]) -> None:
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
                value = torch.cat([value, padding], dim=dim)
            kwargs[key] = value

    def compute(self) -> Dict[str, float]:
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

        metrics: Dict[str, float] = {}
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

    def _build_documents(self) -> Tuple[List[Document], ...]:
        assert self.dataset is not None, "dataset isn't set"
        assert self.reading_id2reading is not None, "reading_id2reading isn't set"

        doc_id2predicted_sentences: Dict[str, List[Sentence]] = defaultdict(list)
        doc_id2partly_gold_sentences1: Dict[str, List[Sentence]] = defaultdict(list)
        doc_id2partly_gold_sentences2: Dict[str, List[Sentence]] = defaultdict(list)
        doc_id2gold_sentences: Dict[str, List[Sentence]] = defaultdict(list)
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
    def _convert_doc_id2sentences_into_documents(doc_id2sentences: Dict[str, List[Sentence]]) -> List[Document]:
        return [Document.from_sentences(ss) for ss in doc_id2sentences.values()]

    @staticmethod
    def compute_reading_prediction_metrics(
        predicted_documents: List[Document], gold_documents: List[Document]
    ) -> Dict[str, float]:
        reading_predictions = [m.reading for d in predicted_documents for m in d.morphemes]
        reading_labels = [m.reading for d in gold_documents for m in d.morphemes]
        num_correct = sum(p == l for p, l in zip(reading_predictions, reading_labels))
        # 単語単位の読みの正解率
        return {"reading_prediction_accuracy": num_correct / len(reading_labels)}

    @staticmethod
    def compute_morphological_analysis_metrics(
        predicted_documents: List[Document], gold_documents: List[Document]
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # m.pos not in POS_TAGS ... 未定義語、その他など
        pos_labels = [[f"B-{m.pos}" if m.pos in POS_TAGS else "B-" for m in d.morphemes] for d in gold_documents]
        pos_predictions = [[f"B-{m.pos}" for m in d.morphemes] for d in predicted_documents]
        metrics["pos_f1"] = f1_score(y_true=pos_labels, y_pred=pos_predictions).item()

        subpos_labels = [
            [f"B-{m.subpos}" if m.subpos in SUBPOS_TAGS else "B-" for m in d.morphemes] for d in gold_documents
        ]
        subpos_predictions = [[f"B-{m.subpos}" for m in d.morphemes] for d in predicted_documents]
        metrics["subpos_f1"] = f1_score(y_true=subpos_labels, y_pred=subpos_predictions).item()

        conjtype_labels = [
            [f"B-{m.conjtype}" if m.conjtype in CONJTYPE_TAGS else "B-" for m in d.morphemes] for d in gold_documents
        ]
        conjtype_predictions = [[f"B-{m.conjtype}" for m in d.morphemes] for d in predicted_documents]
        metrics["conjtype_f1"] = f1_score(y_true=conjtype_labels, y_pred=conjtype_predictions).item()

        conjform_labels = [
            [f"B-{m.conjform}" if m.conjform in CONJFORM_TAGS else "B-" for m in d.morphemes] for d in gold_documents
        ]
        conjform_predictions = [[f"B-{m.conjform}" for m in d.morphemes] for d in predicted_documents]
        metrics["conjform_f1"] = f1_score(y_true=conjform_labels, y_pred=conjform_predictions).item()

        metrics["morphological_analysis_f1"] = mean(metrics.values())
        return metrics

    def compute_word_feature_tagging_metrics(
        self, predicted_documents: List[Document], gold_documents: List[Document]
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        concatenated_labels = []
        concatenated_predictions = []
        for word_feature in WORD_FEATURES:
            if word_feature == "基本句-区切":
                labels = [self._convert_units_into_segmentation_tags(d.base_phrases) for d in gold_documents]
                predictions = [self._convert_units_into_segmentation_tags(d.base_phrases) for d in predicted_documents]
            elif word_feature == "文節-区切":
                labels = [self._convert_units_into_segmentation_tags(d.phrases) for d in gold_documents]
                predictions = [self._convert_units_into_segmentation_tags(d.phrases) for d in predicted_documents]
            else:
                labels = [
                    [self._convert_feature_to_bo_tag(m.features.get(word_feature)) for m in d.morphemes]
                    for d in gold_documents
                ]
                predictions = [
                    [self._convert_feature_to_bo_tag(m.features.get(word_feature)) for m in d.morphemes]
                    for d in predicted_documents
                ]
            metrics[f"{word_feature}_f1"] = f1_score(
                y_true=labels, y_pred=predictions, mode="strict", scheme=IOB2
            ).item()
            concatenated_labels += labels
            concatenated_predictions += predictions
        metrics["macro_word_feature_tagging_f1"] = mean(metrics.values())
        metrics["micro_word_feature_tagging_f1"] = f1_score(
            y_true=concatenated_labels, y_pred=concatenated_predictions, mode="strict", scheme=IOB2
        ).item()
        return metrics

    @staticmethod
    def _convert_units_into_segmentation_tags(units: Union[List[Phrase], List[BasePhrase]]) -> List[str]:
        return ["B" if m == u.morphemes[-1] else "I" for u in units for m in u.morphemes]

    @staticmethod
    def compute_ner_metrics(predicted_documents: List[Document], gold_documents: List[Document]) -> Dict[str, float]:
        for document in predicted_documents + gold_documents:
            for named_entity in document.named_entities:
                for i, morpheme in enumerate(named_entity.morphemes):
                    bi = "B" if i == 0 else "I"
                    morpheme.features["NE"] = f"{bi}-{named_entity.category.value}"
        labels = [[m.features.get("NE") or "O" for m in d.morphemes] for d in gold_documents]
        predictions = [[m.features.get("NE") or "O" for m in d.morphemes] for d in predicted_documents]
        # `f1_score` function calculates micro average by default
        return {"ner_f1": f1_score(y_true=labels, y_pred=predictions, mode="strict", scheme=IOB2).item()}

    @staticmethod
    def compute_base_phrase_feature_tagging_metrics(
        partly_gold_documents1: List[Document], gold_documents: List[Document]
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        concatenated_labels = []
        concatenated_predictions = []
        for base_phrase_feature in BASE_PHRASE_FEATURES:
            labels = [
                [
                    WordModuleMetric._convert_feature_to_bo_tag(base_phrase.features.get(base_phrase_feature))
                    for base_phrase in d.base_phrases
                ]
                for d in gold_documents
            ]
            predictions = [
                [
                    WordModuleMetric._convert_feature_to_bo_tag(base_phrase.features.get(base_phrase_feature))
                    for base_phrase in d.base_phrases
                ]
                for d in partly_gold_documents1
            ]
            # 正解ラベルがない基本句素性は評価対象外
            if "B" not in set(chain.from_iterable(labels)):
                continue
            metrics[f"{base_phrase_feature}_f1"] = f1_score(
                y_true=labels, y_pred=predictions, mode="strict", scheme=IOB2
            ).item()
            concatenated_labels += labels
            concatenated_predictions += predictions
        metrics["macro_base_phrase_feature_tagging_f1"] = mean(metrics.values())
        metrics["micro_base_phrase_feature_tagging_f1"] = f1_score(
            y_true=concatenated_labels, y_pred=concatenated_predictions, mode="strict", scheme=IOB2
        ).item()
        return metrics

    @staticmethod
    def _convert_feature_to_bo_tag(feature: Optional[Union[bool, str]]) -> str:
        return "B" if feature not in (None, False) else "O"

    @staticmethod
    def compute_dependency_parsing_metrics(
        partly_gold_documents1: List[Document], gold_documents: List[Document]
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        metrics.update(
            {
                f"base_phrase_{k}": v
                for k, v in conll18_ud_eval(
                    *WordModuleMetric._to_conll_lines(partly_gold_documents1, gold_documents, "base_phrase")
                ).items()
            }
        )
        metrics.update(
            {
                f"morpheme_{k}": v
                for k, v in conll18_ud_eval(
                    *WordModuleMetric._to_conll_lines(partly_gold_documents1, gold_documents, "morpheme")
                ).items()
            }
        )
        return metrics

    @staticmethod
    def _to_conll_lines(
        partly_gold_documents1: List[Document], gold_documents: List[Document], unit: Literal["base_phrase", "morpheme"]
    ) -> Tuple[List[str], List[str]]:
        gold_lines = []
        system_lines = []
        for gold_document, partly_gold_document1 in zip(gold_documents, partly_gold_documents1):
            for sentence in gold_document.sentences:
                gold_lines += [WordModuleMetric._to_conll_line(u) for u in getattr(sentence, f"{unit}s")]
                gold_lines.append("\n")
            for sentence in partly_gold_document1.sentences:
                system_lines += [WordModuleMetric._to_conll_line(u) for u in getattr(sentence, f"{unit}s")]
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
        self, partly_gold_documents2: List[Document], gold_documents: List[Document]
    ) -> Dict[str, float]:
        assert self.dataset is not None, "dataset isn't set"
        pas_extractor = self.dataset.cohesion_task2extractor[CohesionTask.PAS_ANALYSIS]
        assert isinstance(pas_extractor, PasExtractor), "pas utils isn't set correctly"

        scorer = CohesionScorer(
            exophora_referent_types=self.dataset.exophora_referent_types,
            pas_cases=pas_extractor.cases if CohesionTask.PAS_ANALYSIS in self.dataset.cohesion_tasks else [],
            pas_verbal=pas_extractor.verbal_predicate,
            pas_nominal=pas_extractor.nominal_predicate,
            bridging=(CohesionTask.BRIDGING_REFERENCE_RESOLUTION in self.dataset.cohesion_tasks),
            coreference=(CohesionTask.COREFERENCE_RESOLUTION in self.dataset.cohesion_tasks),
        )
        score: CohesionScore = scorer.run(partly_gold_documents2, gold_documents)

        metrics: Dict[str, float] = {}
        for task, analysis2metric in score.to_dict().items():
            for analysis, metric in analysis2metric.items():
                if task not in ("bridging", "coreference"):
                    key = "pas"
                    if task != "all_case":
                        key += f"_{task}"
                else:
                    key = task
                if analysis != "all":
                    key += f"_{analysis}"
                key += "_f1"
                metrics[key] = metric.f1
        cohesion_analysis_f1s = []
        if CohesionTask.PAS_ANALYSIS in self.dataset.cohesion_tasks:
            cohesion_analysis_f1s.append(metrics["pas_f1"])
        if CohesionTask.BRIDGING_REFERENCE_RESOLUTION in self.dataset.cohesion_tasks:
            cohesion_analysis_f1s.append(metrics["bridging_f1"])
        if CohesionTask.COREFERENCE_RESOLUTION in self.dataset.cohesion_tasks:
            cohesion_analysis_f1s.append(metrics["coreference_f1"])
        metrics["cohesion_analysis_f1"] = mean(cohesion_analysis_f1s) if cohesion_analysis_f1s else 0.0
        return metrics

    @staticmethod
    def compute_discourse_relation_analysis_metrics(
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        ignored_indices = labels.eq(IGNORE_INDEX)
        predictions = predictions[~ignored_indices]
        labels = labels[~ignored_indices]

        if (~ignored_indices).sum().item() == 0:
            accuracy = 0.0
        else:
            accuracy = accuracy_score(y_true=labels, y_pred=predictions).item()

        no_relation_index = DISCOURSE_RELATIONS.index("談話関係なし")

        ignored_indices = predictions.eq(no_relation_index)
        if (~ignored_indices).sum().item() == 0:
            precision = 0.0
        else:
            precision = accuracy_score(y_true=labels[~ignored_indices], y_pred=predictions[~ignored_indices]).item()

        ignored_indices = labels.eq(no_relation_index)
        if (~ignored_indices).sum().item() == 0:
            recall = 0.0
        else:
            recall = accuracy_score(y_true=labels[~ignored_indices], y_pred=predictions[~ignored_indices]).item()

        if (precision + recall) == 0.0:
            f1 = 0.0
        else:
            f1 = (2 * precision * recall) / (precision + recall)

        return {
            "discourse_relation_analysis_accuracy": accuracy,
            "discourse_relation_analysis_precision": precision,
            "discourse_relation_analysis_recall": recall,
            "discourse_relation_analysis_f1": f1,
        }
