import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Union

from omegaconf import ListConfig
from rhoknp import Document, Sentence
from rhoknp.cohesion import ExophoraReferent
from tokenizers import Encoding
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from kwja.datamodule.datasets.base import BaseDataset, FullAnnotatedDocumentLoaderMixin
from kwja.datamodule.examples import WordExample
from kwja.utils.cohesion_analysis import BridgingUtils, CohesionBasePhrase, CohesionUtils, CoreferenceUtils, PasUtils
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
from kwja.utils.progress_bar import track
from kwja.utils.reading_prediction import ReadingAligner, get_reading2reading_id
from kwja.utils.sub_document import extract_target_sentences

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WordModuleFeatures:
    example_ids: int
    input_ids: List[int]
    attention_mask: List[int]
    target_mask: List[int]
    subword_map: List[List[bool]]
    reading_labels: List[int]
    reading_subword_map: List[List[bool]]
    pos_labels: List[int]
    subpos_labels: List[int]
    conjtype_labels: List[int]
    conjform_labels: List[int]
    word_feature_labels: List[List[int]]
    ne_labels: List[int]
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
        super(WordDataset, self).__init__(tokenizer, max_seq_length)
        self.path = Path(path)
        if tokenizer.name_or_path in SPLIT_INTO_WORDS_MODEL_NAMES:
            self.tokenizer_input_format: Literal["words", "text"] = "words"
        else:
            self.tokenizer_input_format = "text"
        super(BaseDataset, self).__init__(self.path, tokenizer, max_seq_length, document_split_stride)
        # ---------- seq2seq ----------
        self.from_seq2seq: bool = False

        # ---------- reading prediction ----------
        reading_resource_path = RESOURCE_PATH / "reading_prediction"
        self.reading2reading_id = get_reading2reading_id(reading_resource_path / "vocab.txt")
        self.reading_aligner = ReadingAligner(
            self.tokenizer, self.tokenizer_input_format, KanjiDic(str(reading_resource_path / "kanjidic"))
        )

        # ---------- cohesion analysis ----------
        self.cohesion_tasks: List[CohesionTask] = [CohesionTask(t) for t in cohesion_tasks]
        self.exophora_referents = [ExophoraReferent(s) for s in exophora_referents]
        self.restrict_cohesion_target = restrict_cohesion_target
        self.pas_cases: List[str] = list(pas_cases)
        self.br_cases: List[str] = list(br_cases)
        self.cohesion_task2utils: Dict[CohesionTask, CohesionUtils] = {}
        for cohesion_task in self.cohesion_tasks:
            if cohesion_task == CohesionTask.PAS_ANALYSIS:
                self.cohesion_task2utils[cohesion_task] = PasUtils(
                    self.pas_cases, "all", self.exophora_referents, restrict_target=restrict_cohesion_target
                )
            elif cohesion_task == CohesionTask.BRIDGING_REFERENCE_RESOLUTION:
                self.cohesion_task2utils[cohesion_task] = BridgingUtils(
                    self.br_cases, self.exophora_referents, restrict_target=restrict_cohesion_target
                )
            elif cohesion_task == CohesionTask.COREFERENCE_RESOLUTION:
                self.cohesion_task2utils[cohesion_task] = CoreferenceUtils(
                    self.exophora_referents, restrict_target=restrict_cohesion_target
                )

        # ---------- dependency parsing & cohesion analysis ----------
        self.special_tokens: List[str] = list(special_tokens)
        self.special_token2index: Dict[str, int] = {
            st: self.max_seq_length - len(self.special_tokens) + i for i, st in enumerate(self.special_tokens)
        }
        self.index2special_token: Dict[int, str] = {v: k for k, v in self.special_token2index.items()}
        self.special_encoding: Encoding = self.tokenizer(
            self.special_tokens,
            is_split_into_words=True,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=False,
            add_special_tokens=False,
        ).encodings[0]

        self.examples: List[WordExample] = self._load_examples(self.documents)

    def _get_tokenized_len(self, document_or_sentence: Union[Document, Sentence]) -> int:
        tokenizer_input: Union[List[str], str] = [m.text for m in document_or_sentence.morphemes]
        if self.tokenizer_input_format == "text":
            tokenizer_input = " ".join(tokenizer_input)
        encoding = self.tokenizer(
            tokenizer_input, add_special_tokens=False, is_split_into_words=self.tokenizer_input_format == "words"
        ).encodings[0]
        return len(encoding.ids)

    def _load_examples(self, documents: List[Document]) -> List[WordExample]:
        examples = []
        example_id = 0
        for document in track(documents, description="Loading examples"):
            tokenizer_input: Union[List[str], str] = [m.text for m in document.morphemes]
            if self.tokenizer_input_format == "text":
                tokenizer_input = " ".join(tokenizer_input)
            encoding: Encoding = self.tokenizer(
                tokenizer_input,
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=False,
                max_length=self.max_seq_length - self.num_special_tokens,
                is_split_into_words=self.tokenizer_input_format == "words",
            ).encodings[0]
            if len(encoding.ids) > self.max_seq_length - self.num_special_tokens:
                continue

            example = WordExample(example_id, encoding)
            example.load_document(document, self.reading_aligner, self.cohesion_task2utils)
            discourse_path = self.path / "disc_expert" / f"{document.doc_id}.knp"
            if not discourse_path.exists() and self.path.name == "train":
                discourse_path = self.path / "disc_crowd" / f"{document.doc_id}.knp"
            if discourse_path.exists():
                try:
                    discourse_document = Document.from_knp(discourse_path.read_text())
                    if document == discourse_document:
                        example.load_discourse_document(discourse_document)
                except AssertionError:
                    logger.warning(f"{discourse_path} is not a valid KNP file")

            examples.append(example)
            example_id += 1
        if len(examples) == 0:
            logger.error(
                "No examples to process. "
                f"Make sure there exist any documents in {self.path} and they are not too long."
            )
        return examples

    def encode(self, example: WordExample) -> WordModuleFeatures:
        merged_encoding: Encoding = Encoding.merge([example.encoding, self.special_encoding])
        assert example.doc_id is not None, "doc_id isn't set"
        document = self.doc_id2document[example.doc_id]

        target_mask = [0 for _ in range(self.max_seq_length)]
        for sentence in extract_target_sentences(document):
            for morpheme in sentence.morphemes:
                target_mask[morpheme.global_index] = 1

        # ---------- reading prediction ----------
        reading_labels = [IGNORE_INDEX] * self.max_seq_length
        if example.readings is not None:
            token_indices = [
                (token_index, word_id)
                for token_index, word_id in enumerate(merged_encoding.word_ids)
                if token_index not in self.index2special_token and word_id is not None
            ]
            for (token_index, word_id), reading in zip(token_indices, example.readings):
                if target_mask[word_id] == 1:
                    decoded_token = self.tokenizer.decode(merged_encoding.ids[token_index])
                    if decoded_token == reading:
                        reading_labels[token_index] = self.reading2reading_id["[ID]"]
                    else:
                        reading_labels[token_index] = self.reading2reading_id.get(
                            reading, self.reading2reading_id["[UNK]"]
                        )

        # NOTE: hereafter, indices are given at the word level

        # ---------- morphological analysis ----------
        morpheme_attribute_tags_list = (POS_TAGS, SUBPOS_TAGS, CONJTYPE_TAGS, CONJFORM_TAGS)
        morpheme_attribute_labels = tuple([IGNORE_INDEX] * self.max_seq_length for _ in morpheme_attribute_tags_list)
        for morpheme_global_index, morpheme_attributes in example.morpheme_global_index2morpheme_attributes.items():
            for i, (morpheme_attribute, morpheme_attribute_tags) in enumerate(
                zip(morpheme_attributes, morpheme_attribute_tags_list)
            ):
                if morpheme_attribute in morpheme_attribute_tags:
                    morpheme_attribute_index = morpheme_attribute_tags.index(morpheme_attribute)
                    morpheme_attribute_labels[i][morpheme_global_index] = morpheme_attribute_index

        # ---------- word feature tagging ----------
        word_feature_labels = [[IGNORE_INDEX] * len(WORD_FEATURES) for _ in range(self.max_seq_length)]
        for morpheme_global_index, word_feature_set in example.morpheme_global_index2word_feature_set.items():
            for i, word_feature in enumerate(WORD_FEATURES):
                word_feature_labels[morpheme_global_index][i] = int(word_feature in word_feature_set)

        # ---------- ner ----------
        ne_labels: List[int] = [
            NE_TAGS.index("O") if target_mask[morpheme_global_index] == 1 else IGNORE_INDEX
            for morpheme_global_index in range(self.max_seq_length)
        ]
        for named_entity in example.named_entities:
            category = named_entity.category.value
            for i, morpheme in enumerate(named_entity.morphemes):
                bi = "B" if i == 0 else "I"
                assert ne_labels[morpheme.global_index] == NE_TAGS.index("O"), f"nested NE found in {example.doc_id}"
                ne_index = NE_TAGS.index(f"{bi}-{category}")
                ne_labels[morpheme.global_index] = ne_index

        # ---------- base phrase feature tagging ----------
        base_phrase_feature_labels = [[IGNORE_INDEX] * len(BASE_PHRASE_FEATURES) for _ in range(self.max_seq_length)]
        for morpheme_global_index, feature_set in example.morpheme_global_index2base_phrase_feature_set.items():
            for i, base_phrase_feature in enumerate(BASE_PHRASE_FEATURES):
                base_phrase_feature_labels[morpheme_global_index][i] = int(base_phrase_feature in feature_set)

        # ---------- dependency parsing ----------
        dependency_labels: List[int] = [IGNORE_INDEX for _ in range(self.max_seq_length)]
        root_index = self.special_token2index["[ROOT]"]
        for morpheme_global_index, dependency in example.morpheme_global_index2dependency.items():
            dependency_labels[morpheme_global_index] = dependency if dependency >= 0 else root_index
        dependency_mask = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        for morpheme_global_index, head_candidates in example.morpheme_global_index2head_candidates.items():
            for head_candidate in head_candidates:
                dependency_mask[morpheme_global_index][head_candidate.global_index] = True
            dependency_mask[morpheme_global_index][root_index] = True
        dependency_type_labels: List[int] = [IGNORE_INDEX for _ in range(self.max_seq_length)]
        for morpheme_global_index, dependency_type in example.morpheme_global_index2dependency_type.items():
            dependency_type_index = DEPENDENCY_TYPES.index(dependency_type)
            dependency_type_labels[morpheme_global_index] = dependency_type_index

        # ---------- cohesion analysis ----------
        cohesion_labels: List[List[List[int]]] = []  # (rel, seq, seq)
        cohesion_mask: List[List[List[bool]]] = []  # (rel, seq, seq)
        for cohesion_task, cohesion_utils in self.cohesion_task2utils.items():
            cohesion_base_phrases = example.cohesion_task2base_phrases[cohesion_task]
            for rel in cohesion_utils.rels:
                rel_labels = self._convert_cohesion_base_phrases_into_rel_labels(cohesion_base_phrases, rel)
                cohesion_labels.append(rel_labels)
            rel_mask = self._convert_cohesion_base_phrases_into_rel_mask(cohesion_base_phrases)
            cohesion_mask.extend([rel_mask] * len(cohesion_utils.rels))

        # ---------- discourse parsing ----------
        discourse_labels = [[IGNORE_INDEX for _ in range(self.max_seq_length)] for _ in range(self.max_seq_length)]
        for (
            modifier_morpheme_global_index,
            head_morpheme_global_index,
        ), relation in example.morpheme_global_indices2discourse_relation.items():
            if relation in DISCOURSE_RELATIONS:
                discourse_index = DISCOURSE_RELATIONS.index(relation)
                discourse_labels[modifier_morpheme_global_index][head_morpheme_global_index] = discourse_index

        return WordModuleFeatures(
            example_ids=example.example_id,
            input_ids=merged_encoding.ids,
            attention_mask=merged_encoding.attention_mask,
            target_mask=target_mask,
            subword_map=self._get_subword_map(merged_encoding),
            reading_labels=reading_labels,
            reading_subword_map=self._get_subword_map(merged_encoding, include_special_tokens=False),
            pos_labels=morpheme_attribute_labels[0],
            subpos_labels=morpheme_attribute_labels[1],
            conjtype_labels=morpheme_attribute_labels[2],
            conjform_labels=morpheme_attribute_labels[3],
            word_feature_labels=word_feature_labels,
            ne_labels=ne_labels,
            base_phrase_feature_labels=base_phrase_feature_labels,
            dependency_labels=dependency_labels,
            dependency_mask=dependency_mask,
            dependency_type_labels=dependency_type_labels,
            cohesion_labels=cohesion_labels,
            cohesion_mask=cohesion_mask,
            discourse_labels=discourse_labels,
        )

    def _get_subword_map(self, encoding: Encoding, include_special_tokens: bool = True) -> List[List[bool]]:
        subword_map = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        for token_id, word_id in enumerate(encoding.word_ids):
            if word_id is None or token_id in self.special_indices:
                continue
            subword_map[word_id][token_id] = True
        if include_special_tokens:
            for special_index in self.special_indices:
                subword_map[special_index][special_index] = True
        return subword_map

    @property
    def cohesion_special_indices(self) -> List[int]:
        return [i for st, i in self.special_token2index.items() if st != "[ROOT]"]

    @property
    def num_special_tokens(self) -> int:
        return len(self.special_tokens)

    @property
    def special_indices(self) -> List[int]:
        return list(self.special_token2index.values())

    def _convert_cohesion_base_phrases_into_rel_labels(
        self,
        cohesion_base_phrases: List[CohesionBasePhrase],
        rel: str,
    ) -> List[List[int]]:
        rel_labels: List[List[int]] = [[0] * self.max_seq_length for _ in range(self.max_seq_length)]
        for cohesion_base_phrase in cohesion_base_phrases:
            if cohesion_base_phrase.is_target is False:
                continue
            assert cohesion_base_phrase.rel2tags is not None, "rel2tags isn't set"
            for tag in cohesion_base_phrase.rel2tags[rel]:
                if tag in self.special_token2index:
                    target_morpheme_global_index = self.special_token2index[tag]
                else:
                    # int(tag) is the base phrase global index of an endophora argument
                    target_morpheme_global_index = cohesion_base_phrases[int(tag)].head.global_index
                rel_labels[cohesion_base_phrase.head.global_index][target_morpheme_global_index] = 1
        return rel_labels

    def _convert_cohesion_base_phrases_into_rel_mask(
        self,
        cohesion_base_phrases: List[CohesionBasePhrase],
    ) -> List[List[bool]]:
        rel_mask = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        for cohesion_base_phrase in cohesion_base_phrases:
            if cohesion_base_phrase.is_target is False:
                continue
            assert cohesion_base_phrase.antecedent_candidates is not None, "antecedent_candidates isn't set"
            for morpheme in cohesion_base_phrase.morphemes:
                for antecedent_candidate in cohesion_base_phrase.antecedent_candidates:
                    rel_mask[morpheme.global_index][antecedent_candidate.head.global_index] = True
                for cohesion_special_index in self.cohesion_special_indices:
                    rel_mask[morpheme.global_index][cohesion_special_index] = True
        return rel_mask
