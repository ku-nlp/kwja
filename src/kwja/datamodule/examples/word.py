import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Optional, Set, Tuple

from cohesion_tools.extractors.base import BaseExtractor
from rhoknp import BasePhrase, Clause, Document, Morpheme, Phrase
from rhoknp.props import DepType, NamedEntity
from tokenizers import Encoding

from kwja.utils.cohesion_analysis import CohesionBasePhrase, wrap_base_phrases
from kwja.utils.constants import (
    BASE_PHRASE_FEATURES,
    DISCOURSE_RELATION_MAP,
    IGNORE_VALUE_FEATURE_PAT,
    SUB_WORD_FEATURES,
    CohesionTask,
)
from kwja.utils.reading_prediction import ReadingAligner
from kwja.utils.sub_document import extract_target_sentences

logger = logging.getLogger(__name__)


class SpecialTokenIndexer:
    def __init__(self, special_tokens: List[str], num_tokens: int, num_morphemes: int) -> None:
        self.special_tokens: List[str] = special_tokens
        self._special_token2token_level_index: Dict[str, int] = {
            st: num_tokens + i for i, st in enumerate(special_tokens)
        }
        self._special_token2morpheme_level_index: Dict[str, int] = {
            st: num_morphemes + i for i, st in enumerate(special_tokens)
        }

    def get_morpheme_level_index(self, special_token: str) -> int:
        return self._special_token2morpheme_level_index[special_token]

    def get_morpheme_level_indices(self, only_cohesion: bool = False) -> List[int]:
        return [
            i for st, i in self._special_token2morpheme_level_index.items() if not (only_cohesion and st == "[ROOT]")
        ]

    @cached_property
    def token_level_indices(self) -> List[int]:
        return list(self._special_token2token_level_index.values())

    @cached_property
    def token_and_morpheme_level_indices(self) -> List[Tuple[int, int]]:
        return list(
            zip(self._special_token2token_level_index.values(), self._special_token2morpheme_level_index.values())
        )


class WordExample:
    def __init__(self, example_id: int, encoding: Encoding, special_token_indexer: SpecialTokenIndexer) -> None:
        self.example_id = example_id
        self.encoding = encoding
        self.special_token_indexer = special_token_indexer
        self.doc_id: Optional[str] = None
        self.analysis_target_morpheme_indices: List[int] = []

        # ---------- reading prediction ----------
        self.readings: Optional[List[str]] = None

        # ---------- morphological analysis ----------
        self.morpheme_global_index2morpheme_attributes: Dict[int, Tuple[str, str, str, str]] = {}

        # ---------- word feature tagging ----------
        self.morpheme_global_index2word_feature_set: Dict[int, Set[str]] = {}

        # ---------- ner ----------
        self.named_entities: List[NamedEntity] = []

        # ---------- base phrase feature tagging ----------
        self.morpheme_global_index2base_phrase_feature_set: Dict[int, Set[str]] = {}

        # ---------- dependency parsing ----------
        self.morpheme_global_index2dependency: Dict[int, int] = {}
        # 形態素単位係り先候補
        self.morpheme_global_index2head_candidates: Dict[int, List[int]] = {}
        self.morpheme_global_index2dependency_type: Dict[int, DepType] = {}

        # ---------- cohesion analysis ----------
        # wrapされた基本句のリスト
        self.cohesion_task2base_phrases: Dict[CohesionTask, List[CohesionBasePhrase]] = {}

        # ---------- discourse relation analysis ----------
        # (modifier morpheme global index, head morpheme global index) -> 談話関係
        self.morpheme_global_indices2discourse_relation: Dict[Tuple[int, int], str] = {}

    def load_document(
        self,
        document: Document,
        reading_aligner: ReadingAligner,
        cohesion_task2extractor: Dict[CohesionTask, BaseExtractor],
        cohesion_task2rels: Dict[CohesionTask, List[str]],
        restrict_cohesion_target: bool,
    ) -> None:
        self.doc_id = document.doc_id
        self._set_readings(document.morphemes, reading_aligner)
        for sentence in extract_target_sentences(document):
            morphemes = sentence.morphemes
            base_phrases = sentence.base_phrases
            self.analysis_target_morpheme_indices += [m.global_index for m in morphemes]
            self._set_morpheme_attributes(morphemes)
            self._set_word_feature_set(sentence.phrases, base_phrases, morphemes)
            self._set_named_entities(sentence.named_entities)
            self._set_base_phrase_feature_set(base_phrases)
            self._set_dependencies(morphemes)
        self._set_cohesion_base_phrases(
            document.base_phrases, cohesion_task2extractor, cohesion_task2rels, restrict_cohesion_target
        )

    def load_discourse_document(self, discourse_document: Document) -> None:
        self._set_discourse_relation(discourse_document.clauses)

    def _set_readings(self, morphemes: List[Morpheme], reading_aligner: ReadingAligner) -> None:
        try:
            self.readings = reading_aligner.align(morphemes)
        except Exception as e:
            logger.warning(e)

    def _set_morpheme_attributes(self, morphemes: List[Morpheme]) -> None:
        for morpheme in morphemes:
            self.morpheme_global_index2morpheme_attributes[morpheme.global_index] = (
                morpheme.pos,
                morpheme.subpos,
                morpheme.conjtype,
                morpheme.conjform,
            )

    def _set_word_feature_set(
        self, phrases: List[Phrase], base_phrases: List[BasePhrase], morphemes: List[Morpheme]
    ) -> None:
        for morpheme in morphemes:
            self.morpheme_global_index2word_feature_set.setdefault(morpheme.global_index, set())
            for sub_word_feature in SUB_WORD_FEATURES:
                if sub_word_feature in morpheme.features:
                    self.morpheme_global_index2word_feature_set[morpheme.global_index].add(sub_word_feature)

        for base_phrase in base_phrases:
            self.morpheme_global_index2word_feature_set[base_phrase.head.global_index].add("基本句-主辞")
            self.morpheme_global_index2word_feature_set[base_phrase.morphemes[-1].global_index].add("基本句-区切")

        for phrase in phrases:
            self.morpheme_global_index2word_feature_set[phrase.morphemes[-1].global_index].add("文節-区切")

    def _set_named_entities(self, named_entities: List[NamedEntity]) -> None:
        self.named_entities += named_entities

    def _set_base_phrase_feature_set(self, base_phrases: List[BasePhrase]) -> None:
        target_base_phrase_feature_set = set(BASE_PHRASE_FEATURES)
        for base_phrase in base_phrases:
            base_phrase_feature_set = {
                k + (f":{v}" if isinstance(v, str) and IGNORE_VALUE_FEATURE_PAT.match(k) is None else "")
                for k, v in base_phrase.features.items()
            }
            self.morpheme_global_index2base_phrase_feature_set[base_phrase.head.global_index] = (
                base_phrase_feature_set & target_base_phrase_feature_set
            )

    def _set_dependencies(self, morphemes: List[Morpheme]) -> None:
        for morpheme in morphemes:
            dependency = morpheme.parent.global_index if morpheme.parent is not None else -1
            self.morpheme_global_index2dependency[morpheme.global_index] = dependency
            self.morpheme_global_index2head_candidates[morpheme.global_index] = [
                m.global_index for m in morphemes if m.global_index != morpheme.global_index
            ]

            if morpheme == morpheme.base_phrase.head:
                assert morpheme.base_phrase.dep_type is not None
                dependency_type = morpheme.base_phrase.dep_type
            else:
                dependency_type = DepType.DEPENDENCY
            self.morpheme_global_index2dependency_type[morpheme.global_index] = dependency_type

    def _set_cohesion_base_phrases(
        self,
        base_phrases: List[BasePhrase],
        cohesion_task2extractor: Dict[CohesionTask, BaseExtractor],
        cohesion_task2rels: Dict[CohesionTask, List[str]],
        restrict_cohesion_target: bool,
    ) -> None:
        for cohesion_task, cohesion_extractor in cohesion_task2extractor.items():
            cohesion_rels = cohesion_task2rels[cohesion_task]
            self.cohesion_task2base_phrases[cohesion_task] = wrap_base_phrases(
                base_phrases, cohesion_extractor, cohesion_rels, restrict_cohesion_target
            )

    def _set_discourse_relation(self, clauses: List[Clause]) -> None:
        for modifier in clauses:
            modifier_morpheme_global_index = modifier.end.morphemes[0].global_index
            for head in clauses:
                head_morpheme_global_index = head.end.morphemes[0].global_index
                key = (modifier_morpheme_global_index, head_morpheme_global_index)
                for discourse_relation in modifier.discourse_relations:
                    if discourse_relation.head == head:
                        label = DISCOURSE_RELATION_MAP[discourse_relation.label.value]
                        self.morpheme_global_indices2discourse_relation[key] = label
                if key not in self.morpheme_global_indices2discourse_relation:
                    self.morpheme_global_indices2discourse_relation[key] = "談話関係なし"


@dataclass(frozen=True)
class WordInferenceExample:
    example_id: int
    encoding: Encoding
    special_token_indexer: SpecialTokenIndexer
    doc_id: str
    analysis_target_morpheme_indices: List[int]
