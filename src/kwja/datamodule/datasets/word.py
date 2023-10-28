import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from cohesion_tools.extractors import BridgingExtractor, CoreferenceExtractor, PasExtractor
from cohesion_tools.extractors.base import BaseExtractor
from omegaconf import ListConfig
from rhoknp import Document, Sentence
from rhoknp.cohesion import ExophoraReferent, ExophoraReferentType
from tokenizers import Encoding
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from kwja.datamodule.datasets.base import BaseDataset, FullAnnotatedDocumentLoaderMixin
from kwja.datamodule.examples import SpecialTokenIndexer, WordExample
from kwja.utils.cohesion_analysis import CohesionBasePhrase
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
    SPLIT_INTO_WORDS_MODEL_NAMES,
    SUBPOS_TAGS,
    WORD_FEATURES,
    CohesionTask,
)
from kwja.utils.kanjidic import KanjiDic
from kwja.utils.logging_util import track
from kwja.utils.reading_prediction import ReadingAligner, get_reading2reading_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WordModuleFeatures:
    example_ids: int
    input_ids: List[int]
    attention_mask: List[int]
    special_token_indices: List[int]
    subword_map: List[List[bool]]
    reading_labels: List[int]
    reading_subword_map: List[List[bool]]
    pos_labels: List[int]
    subpos_labels: List[int]
    conjtype_labels: List[int]
    conjform_labels: List[int]
    word_feature_labels: List[List[int]]
    ne_labels: List[int]
    ne_mask: List[bool]
    base_phrase_feature_labels: List[List[int]]
    dependency_labels: List[int]
    dependency_mask: List[List[bool]]  # True/False = keep/mask
    dependency_type_labels: List[int]
    cohesion_labels: List[List[List[int]]]
    cohesion_mask: List[List[List[bool]]]  # True/False = keep/mask
    discourse_labels: List[List[int]]


class WordDataset(BaseDataset[WordExample, WordModuleFeatures], FullAnnotatedDocumentLoaderMixin):
    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        document_split_stride: int,
        cohesion_tasks: ListConfig,
        exophora_referents: ListConfig,
        restrict_cohesion_target: bool,
        pas_cases: ListConfig,
        br_cases: ListConfig,
        special_tokens: ListConfig,
    ) -> None:
        super().__init__(tokenizer, max_seq_length)
        self.path = Path(path)
        if tokenizer.name_or_path in SPLIT_INTO_WORDS_MODEL_NAMES:
            self.tokenizer_input_format: Literal["words", "text"] = "words"
        else:
            self.tokenizer_input_format = "text"
        super(BaseDataset, self).__init__(self.path, tokenizer, max_seq_length, document_split_stride)
        # some tags are not annotated in editorial articles
        self.skip_cohesion_ne_discourse = self.path.parts[-2] == "kyoto_ed"

        # ---------- reading prediction ----------
        reading_resource_path = RESOURCE_PATH / "reading_prediction"
        self.reading2reading_id = get_reading2reading_id(reading_resource_path / "vocab.txt")
        self.reading_aligner = ReadingAligner(
            self.tokenizer, self.tokenizer_input_format, KanjiDic(str(reading_resource_path / "kanjidic"))
        )

        # ---------- cohesion analysis ----------
        self.cohesion_tasks: List[CohesionTask] = [task for task in CohesionTask if task.value in cohesion_tasks]
        self.exophora_referent_types: List[ExophoraReferentType] = [
            ExophoraReferent(er).type for er in exophora_referents
        ]
        self.cohesion_task2extractor: Dict[CohesionTask, BaseExtractor] = {
            CohesionTask.PAS_ANALYSIS: PasExtractor(
                list(pas_cases),
                self.exophora_referent_types,
                verbal_predicate=True,
                nominal_predicate=True,
            ),
            CohesionTask.BRIDGING_REFERENCE_RESOLUTION: BridgingExtractor(list(br_cases), self.exophora_referent_types),
            CohesionTask.COREFERENCE_RESOLUTION: CoreferenceExtractor(self.exophora_referent_types),
        }
        self.cohesion_task2rels: Dict[CohesionTask, List[str]] = {
            CohesionTask.PAS_ANALYSIS: list(pas_cases),
            CohesionTask.BRIDGING_REFERENCE_RESOLUTION: list(br_cases),
            CohesionTask.COREFERENCE_RESOLUTION: ["="],
        }
        self.restrict_cohesion_target: bool = restrict_cohesion_target

        # ---------- dependency parsing & cohesion analysis ----------
        self.special_tokens: List[str] = list(special_tokens)
        self.special_encoding: Encoding = self.tokenizer(
            self.special_tokens,
            add_special_tokens=False,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=False,
            is_split_into_words=True,
        ).encodings[0]

        self.examples: List[WordExample] = self._load_examples(self.doc_id2document)
        is_training = self.path.parts[-1] == "train" or (
            self.path.parts[-2] == "kyoto_ed" and self.path.parts[-1] == "all"
        )
        if is_training is True:
            del self.doc_id2document  # for saving memory

    def _get_tokenized_len(self, document_or_sentence: Union[Document, Sentence]) -> int:
        tokenizer_input: Union[List[str], str] = [m.text for m in document_or_sentence.morphemes]
        if self.tokenizer_input_format == "text":
            tokenizer_input = " ".join(tokenizer_input)
        return len(self.tokenizer.tokenize(tokenizer_input, is_split_into_words=self.tokenizer_input_format == "words"))

    def _load_examples(self, doc_id2document: Dict[str, Document]) -> List[WordExample]:
        examples = []
        example_id = 0
        for document in track(doc_id2document.values(), description="Loading examples"):
            tokenizer_input: Union[List[str], str] = [m.text for m in document.morphemes]
            if self.tokenizer_input_format == "text":
                tokenizer_input = " ".join(tokenizer_input)
            encoding: Encoding = self.tokenizer(
                tokenizer_input,
                padding=PaddingStrategy.DO_NOT_PAD,
                truncation=False,
                is_split_into_words=self.tokenizer_input_format == "words",
            ).encodings[0]
            if len(encoding.ids) > self.max_seq_length - len(self.special_tokens):
                continue
            padding_encoding: Encoding = self.tokenizer(
                "",
                add_special_tokens=False,
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=False,
                max_length=self.max_seq_length - len(encoding.ids) - len(self.special_tokens),
            ).encodings[0]
            merged_encoding: Encoding = Encoding.merge([encoding, self.special_encoding, padding_encoding])

            special_token_indexer = SpecialTokenIndexer(self.special_tokens, len(encoding.ids), len(document.morphemes))

            example = WordExample(example_id, merged_encoding, special_token_indexer)
            example.load_document(
                document,
                self.reading_aligner,
                self.cohesion_task2extractor,
                self.cohesion_task2rels,
                self.restrict_cohesion_target,
            )
            if discourse_document := self._find_discourse_document(document):
                example.load_discourse_document(discourse_document)

            examples.append(example)
            example_id += 1
        if len(examples) == 0:
            logger.error(
                "No examples to process. "
                f"Make sure there exist any documents in {self.path} and they are not too long."
            )
        return examples

    def encode(self, example: WordExample) -> WordModuleFeatures:
        assert example.doc_id is not None, "doc_id isn't set"

        target_mask = [False] * self.max_seq_length
        for global_index in example.analysis_target_morpheme_indices:
            target_mask[global_index] = True

        # ---------- reading prediction ----------
        reading_labels = [IGNORE_INDEX] * self.max_seq_length
        if example.readings is not None:
            token_indices = [
                (token_index, word_id)
                for token_index, word_id in enumerate(example.encoding.word_ids)
                if token_index not in example.special_token_indexer.token_level_indices and word_id is not None
            ]
            for (token_index, word_id), reading in zip(token_indices, example.readings):
                if target_mask[word_id] is False:
                    continue
                decoded_token = self.tokenizer.decode(example.encoding.ids[token_index])
                if decoded_token == reading:
                    reading_labels[token_index] = self.reading2reading_id["[ID]"]
                else:
                    reading_labels[token_index] = self.reading2reading_id.get(reading, self.reading2reading_id["[UNK]"])

        # NOTE: hereafter, indices are given at the word level

        # ---------- morphological analysis ----------
        morpheme_attribute_tags_set = (POS_TAGS, SUBPOS_TAGS, CONJTYPE_TAGS, CONJFORM_TAGS)
        morpheme_attribute_labels_set = tuple([IGNORE_INDEX] * self.max_seq_length for _ in morpheme_attribute_tags_set)
        for morpheme_global_index, morpheme_attributes in example.morpheme_global_index2morpheme_attributes.items():
            for morpheme_attribute, morpheme_attribute_tags, morpheme_attribute_labels in zip(
                morpheme_attributes, morpheme_attribute_tags_set, morpheme_attribute_labels_set
            ):
                if morpheme_attribute not in morpheme_attribute_tags:
                    continue
                morpheme_attribute_labels[morpheme_global_index] = morpheme_attribute_tags.index(morpheme_attribute)

        # ---------- word feature tagging ----------
        word_feature_labels = [[IGNORE_INDEX] * len(WORD_FEATURES) for _ in range(self.max_seq_length)]
        for morpheme_global_index, word_feature_set in example.morpheme_global_index2word_feature_set.items():
            for i, word_feature in enumerate(WORD_FEATURES):
                word_feature_labels[morpheme_global_index][i] = int(word_feature in word_feature_set)

        # ---------- ner ----------
        ne_mask = [False] * self.max_seq_length if self.skip_cohesion_ne_discourse is True else target_mask
        ne_labels: List[int] = [NE_TAGS.index("O")] * self.max_seq_length
        for named_entity in example.named_entities:
            category = named_entity.category.value
            for i, morpheme in enumerate(named_entity.morphemes):
                bi = "B" if i == 0 else "I"
                assert ne_labels[morpheme.global_index] == NE_TAGS.index("O"), f"nested NE found in {example.doc_id}"
                ne_labels[morpheme.global_index] = NE_TAGS.index(f"{bi}-{category}")

        # ---------- base phrase feature tagging ----------
        base_phrase_feature_labels = [[IGNORE_INDEX] * len(BASE_PHRASE_FEATURES) for _ in range(self.max_seq_length)]
        for morpheme_global_index, feature_set in example.morpheme_global_index2base_phrase_feature_set.items():
            for i, base_phrase_feature in enumerate(BASE_PHRASE_FEATURES):
                base_phrase_feature_labels[morpheme_global_index][i] = int(base_phrase_feature in feature_set)

        # ---------- dependency parsing ----------
        dependency_labels: List[int] = [IGNORE_INDEX] * self.max_seq_length
        root_index = example.special_token_indexer.get_morpheme_level_index("[ROOT]")
        for morpheme_global_index, dependency in example.morpheme_global_index2dependency.items():
            dependency_labels[morpheme_global_index] = root_index if dependency == -1 else dependency
        dependency_mask = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        for morpheme_global_index, head_candidates in example.morpheme_global_index2head_candidates.items():
            for head_candidate_global_index in head_candidates:
                dependency_mask[morpheme_global_index][head_candidate_global_index] = True
            dependency_mask[morpheme_global_index][root_index] = True
        dependency_type_labels: List[int] = [IGNORE_INDEX] * self.max_seq_length
        for morpheme_global_index, dependency_type in example.morpheme_global_index2dependency_type.items():
            dependency_type_labels[morpheme_global_index] = DEPENDENCY_TYPES.index(dependency_type)

        # ---------- cohesion analysis ----------
        cohesion_labels: List[List[List[int]]] = []  # (rel, seq, seq)
        cohesion_mask: List[List[List[bool]]] = []  # (rel, seq, seq)
        for cohesion_task in self.cohesion_tasks:
            cohesion_rels = self.cohesion_task2rels[cohesion_task]
            cohesion_base_phrases = example.cohesion_task2base_phrases[cohesion_task]
            for rel in cohesion_rels:
                rel_labels = self._convert_cohesion_base_phrases_into_rel_labels(
                    cohesion_base_phrases, rel, example.special_token_indexer
                )
                cohesion_labels.append(rel_labels)
            rel_mask = self._convert_cohesion_base_phrases_into_rel_mask(
                cohesion_base_phrases, example.special_token_indexer
            )
            cohesion_mask.extend([rel_mask] * len(cohesion_rels))

        # ---------- discourse relation analysis ----------
        discourse_labels = [[IGNORE_INDEX] * self.max_seq_length for _ in range(self.max_seq_length)]
        if self.skip_cohesion_ne_discourse is False:
            for (
                morpheme_global_indices,
                discourse_relation,
            ) in example.morpheme_global_indices2discourse_relation.items():
                if discourse_relation not in DISCOURSE_RELATIONS:
                    continue
                modifier_morpheme_global_index, head_morpheme_global_index = morpheme_global_indices
                discourse_index = DISCOURSE_RELATIONS.index(discourse_relation)
                discourse_labels[modifier_morpheme_global_index][head_morpheme_global_index] = discourse_index

        return WordModuleFeatures(
            example_ids=example.example_id,
            input_ids=example.encoding.ids,
            attention_mask=example.encoding.attention_mask,
            special_token_indices=example.special_token_indexer.token_level_indices,
            subword_map=self._generate_subword_map(example.encoding.word_ids, example.special_token_indexer),
            reading_labels=reading_labels,
            reading_subword_map=self._generate_subword_map(
                example.encoding.word_ids, example.special_token_indexer, include_special_tokens=False
            ),
            pos_labels=morpheme_attribute_labels_set[0],
            subpos_labels=morpheme_attribute_labels_set[1],
            conjtype_labels=morpheme_attribute_labels_set[2],
            conjform_labels=morpheme_attribute_labels_set[3],
            word_feature_labels=word_feature_labels,
            ne_labels=ne_labels,
            ne_mask=ne_mask,
            base_phrase_feature_labels=base_phrase_feature_labels,
            dependency_labels=dependency_labels,
            dependency_mask=dependency_mask,
            dependency_type_labels=dependency_type_labels,
            cohesion_labels=cohesion_labels,
            cohesion_mask=cohesion_mask,
            discourse_labels=discourse_labels,
        )

    def _generate_subword_map(
        self,
        word_ids: List[Union[int, None]],
        special_token_indexer: SpecialTokenIndexer,
        include_special_tokens: bool = True,
    ) -> List[List[bool]]:
        subword_map = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        for token_index, word_id in enumerate(word_ids):
            if word_id is None or token_index in special_token_indexer.token_level_indices:
                continue
            subword_map[word_id][token_index] = True
        if include_special_tokens is True:
            for token_index, morpheme_global_index in special_token_indexer.token_and_morpheme_level_indices:
                subword_map[morpheme_global_index][token_index] = True
        return subword_map

    def _find_discourse_document(self, document: Document) -> Optional[Document]:
        discourse_path = self.path / "disc_expert" / f"{document.doc_id}.knp"
        if not discourse_path.exists() and self.path.name == "train":
            discourse_path = self.path / "disc_crowd" / f"{document.doc_id}.knp"
        if discourse_path.exists():
            try:
                discourse_document = Document.from_knp(discourse_path.read_text())
                if document == discourse_document:
                    return discourse_document
            except AssertionError:
                logger.warning(f"{discourse_path} is not a valid KNP file")
        return None

    def _convert_cohesion_base_phrases_into_rel_labels(
        self,
        cohesion_base_phrases: List[CohesionBasePhrase],
        rel: str,
        special_token_indexer: SpecialTokenIndexer,
    ) -> List[List[int]]:
        rel_labels = [[0] * self.max_seq_length for _ in range(self.max_seq_length)]
        if self.skip_cohesion_ne_discourse is True:
            return rel_labels
        for cohesion_base_phrase in cohesion_base_phrases:
            if cohesion_base_phrase.is_target is False:
                continue
            assert cohesion_base_phrase.rel2tags is not None, "rel2tags isn't set"
            for tag in cohesion_base_phrase.rel2tags[rel]:
                if tag in self.special_tokens:
                    target_morpheme_global_index = special_token_indexer.get_morpheme_level_index(tag)
                else:
                    # int(tag) is the base phrase global index of an endophora argument
                    target_morpheme_global_index = cohesion_base_phrases[int(tag)].head_morpheme_global_index
                rel_labels[cohesion_base_phrase.head_morpheme_global_index][target_morpheme_global_index] = 1
        return rel_labels

    def _convert_cohesion_base_phrases_into_rel_mask(
        self,
        cohesion_base_phrases: List[CohesionBasePhrase],
        special_token_indexer: SpecialTokenIndexer,
    ) -> List[List[bool]]:
        rel_mask = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        for cohesion_base_phrase in cohesion_base_phrases:
            if cohesion_base_phrase.is_target is False:
                continue
            assert cohesion_base_phrase.antecedent_candidates is not None, "antecedent_candidates isn't set"
            for morpheme_global_index in cohesion_base_phrase.morpheme_global_indices:
                for antecedent_candidate in cohesion_base_phrase.antecedent_candidates:
                    rel_mask[morpheme_global_index][antecedent_candidate.head_morpheme_global_index] = True
                for special_token_global_index in special_token_indexer.get_morpheme_level_indices(only_cohesion=True):
                    rel_mask[morpheme_global_index][special_token_global_index] = True
        return rel_mask
