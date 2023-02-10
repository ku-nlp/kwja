import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union

import torch
from omegaconf import ListConfig
from rhoknp import Document, Sentence
from rhoknp.cohesion import ExophoraReferent
from tokenizers import Encoding
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from kwja.datamodule.datasets.base_dataset import BaseDataset
from kwja.datamodule.examples import (
    BasePhraseFeatureExample,
    CohesionExample,
    CohesionTask,
    DependencyExample,
    DiscourseExample,
    ReadingExample,
    WordFeatureExample,
)
from kwja.datamodule.extractors import BridgingExtractor, CoreferenceExtractor, PasExtractor
from kwja.datamodule.extractors.base import Phrase
from kwja.utils.constants import (
    BASE_PHRASE_FEATURES,
    CONJFORM_TAGS,
    CONJTYPE_TAGS,
    DEPENDENCY_TYPES,
    DISCOURSE_RELATIONS,
    IGNORE_INDEX,
    NE_TAGS,
    POS_TAGS,
    SPLIT_INTO_WORDS_MODEL_NAMES,
    SUBPOS_TAGS,
    WORD_FEATURES,
)
from kwja.utils.kanjidic import KanjiDic
from kwja.utils.progress_bar import track
from kwja.utils.reading_prediction import ReadingAligner, get_reading2reading_id
from kwja.utils.sub_document import extract_target_sentences

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WordExampleSet:
    example_id: int
    doc_id: str
    encoding: Encoding
    reading_example: ReadingExample
    word_feature_example: WordFeatureExample
    base_phrase_feature_example: BasePhraseFeatureExample
    dependency_example: DependencyExample
    cohesion_example: CohesionExample
    discourse_example: DiscourseExample


class WordDataset(BaseDataset):
    def __init__(
        self,
        path: str,
        document_split_stride: int,
        reading_resource_path: str,
        pas_cases: ListConfig,
        bar_rels: ListConfig,
        exophora_referents: ListConfig,
        cohesion_tasks: ListConfig,
        restrict_cohesion_target: bool,
        special_tokens: ListConfig,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 512,
    ) -> None:
        self.path = Path(path)
        if tokenizer.name_or_path in SPLIT_INTO_WORDS_MODEL_NAMES:
            self.tokenizer_input_format: Literal["words", "text"] = "words"
        else:
            self.tokenizer_input_format = "text"

        super().__init__(self.path, tokenizer, max_seq_length, document_split_stride)

        # ---------- reading prediction ----------
        self.reading_resource_path = Path(reading_resource_path)
        self.reading2reading_id = get_reading2reading_id(str(self.reading_resource_path / "vocab.txt"))
        self.reading_aligner = ReadingAligner(
            self.tokenizer, self.tokenizer_input_format, KanjiDic(str(self.reading_resource_path / "kanjidic"))
        )

        # ---------- cohesion analysis ----------
        self.pas_cases: List[str] = list(pas_cases)
        self.bar_rels: List[str] = list(bar_rels)
        self.exophora_referents = [ExophoraReferent(s) for s in exophora_referents]
        self.cohesion_tasks: List[CohesionTask] = [CohesionTask(t) for t in cohesion_tasks]
        self.cohesion_task_to_rel_types = {
            CohesionTask.PAS_ANALYSIS: self.pas_cases,
            CohesionTask.BRIDGING: self.bar_rels,
            CohesionTask.COREFERENCE: ["="],
        }
        self.restrict_cohesion_target = restrict_cohesion_target
        self.extractors = {
            CohesionTask.PAS_ANALYSIS: PasExtractor(self.pas_cases, self.exophora_referents, restrict_cohesion_target),
            CohesionTask.COREFERENCE: CoreferenceExtractor(self.exophora_referents, restrict_cohesion_target),
            CohesionTask.BRIDGING: BridgingExtractor(self.bar_rels, self.exophora_referents, restrict_cohesion_target),
        }

        # ---------- dependency parsing & cohesion analysis ----------
        self.special_tokens: List[str] = list(special_tokens)
        self.special_to_index: Dict[str, int] = {
            token: self.max_seq_length - len(self.special_tokens) + i for i, token in enumerate(self.special_tokens)
        }
        self.index_to_special: Dict[int, str] = {v: k for k, v in self.special_to_index.items()}
        self.special_encoding: Encoding = self.tokenizer(
            self.special_tokens,
            is_split_into_words=True,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=False,
            add_special_tokens=False,
        ).encodings[0]

        self.examples: List[WordExampleSet] = self._load_examples(self.documents)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.encode(self.examples[index])

    def _get_tokenized_len(self, document_or_sentence: Union[Document, Sentence]) -> int:
        tokenizer_input: Union[List[str], str] = [m.text for m in document_or_sentence.morphemes]
        if self.tokenizer_input_format == "text":
            tokenizer_input = " ".join(tokenizer_input)
        encoding = self.tokenizer(
            tokenizer_input, add_special_tokens=False, is_split_into_words=self.tokenizer_input_format == "words"
        ).encodings[0]
        return len(encoding.ids)

    def _load_examples(self, documents: List[Document]) -> List[WordExampleSet]:
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

            reading_example = ReadingExample()
            reading_example.load(document, aligner=self.reading_aligner)

            word_feature_example = WordFeatureExample()
            word_feature_example.load(document)

            base_phrase_feature_example = BasePhraseFeatureExample()
            base_phrase_feature_example.load(document)

            dependency_example = DependencyExample()
            dependency_example.load(document)

            cohesion_example = CohesionExample()
            cohesion_example.load(document, tasks=self.cohesion_tasks, extractors=self.extractors)

            discourse_example = DiscourseExample()
            discourse_example.load(document, has_annotation=False)
            path = self.path / "disc_expert" / f"{document.doc_id}.knp"
            if not path.exists() and self.path.name == "train":
                path = self.path / "disc_crowd" / f"{document.doc_id}.knp"
            if path.exists():
                try:
                    discourse_document = Document.from_knp(path.read_text())
                    if document == discourse_document:
                        discourse_example.load(discourse_document)
                except AssertionError:
                    logger.warning(f"{path} is not a valid KNP file")

            examples.append(
                WordExampleSet(
                    example_id=example_id,
                    doc_id=document.doc_id,
                    encoding=encoding,
                    reading_example=reading_example,
                    word_feature_example=word_feature_example,
                    base_phrase_feature_example=base_phrase_feature_example,
                    dependency_example=dependency_example,
                    cohesion_example=cohesion_example,
                    discourse_example=discourse_example,
                )
            )
            example_id += 1
        if len(examples) == 0:
            logger.error(
                "No examples to process. "
                f"Make sure there exist any documents in {self.path} and they are not too long."
            )
        return examples

    def encode(self, example: WordExampleSet) -> Dict[str, torch.Tensor]:
        merged_encoding: Encoding = Encoding.merge([example.encoding, self.special_encoding])
        document = self.doc_id2document[example.doc_id]

        target_mask = [0 for _ in range(self.max_seq_length)]
        for sentence in extract_target_sentences(document):
            for morpheme in sentence.morphemes:
                target_mask[morpheme.global_index] = 1

        # ---------- reading prediction ----------
        reading_example = example.reading_example
        reading_labels = [IGNORE_INDEX] * self.max_seq_length
        if reading_example.readings:
            non_special_token_indices = [
                token_index
                for token_index, word_id in enumerate(merged_encoding.word_ids)
                if word_id is not None and token_index not in self.index_to_special
            ]
            for token_index, non_special_token_index in enumerate(non_special_token_indices):
                reading = reading_example.readings[token_index]
                decoded_token = self.tokenizer.decode(merged_encoding.ids[non_special_token_index])
                if reading == decoded_token:
                    reading_labels[non_special_token_index] = self.reading2reading_id["[ID]"]
                else:
                    reading_labels[non_special_token_index] = self.reading2reading_id.get(
                        reading, self.reading2reading_id["[UNK]"]
                    )

        # NOTE: hereafter, indices are given at the word level

        # ---------- morphological analysis ----------
        word_feature_example = example.word_feature_example
        morpheme_attribute_tags_list = (POS_TAGS, SUBPOS_TAGS, CONJTYPE_TAGS, CONJFORM_TAGS)
        morpheme_attribute_labels = torch.tensor(
            [[IGNORE_INDEX] * len(morpheme_attribute_tags_list) for _ in range(self.max_seq_length)], dtype=torch.long
        )
        for morpheme_global_index, morpheme_attributes in word_feature_example.global_index2attributes.items():
            for i, (morpheme_attribute, morpheme_attribute_tags) in enumerate(
                zip(morpheme_attributes, morpheme_attribute_tags_list)
            ):
                if morpheme_attribute in morpheme_attribute_tags:
                    morpheme_attribute_index = morpheme_attribute_tags.index(morpheme_attribute)
                    morpheme_attribute_labels[morpheme_global_index][i] = morpheme_attribute_index

        # ---------- word feature tagging ----------
        word_feature_labels = [[IGNORE_INDEX] * len(WORD_FEATURES) for _ in range(self.max_seq_length)]
        for morpheme_global_index, word_feature_set in word_feature_example.global_index2feature_set.items():
            for i, word_feature in enumerate(WORD_FEATURES):
                word_feature_labels[morpheme_global_index][i] = int(word_feature in word_feature_set)

        # ---------- ner ----------
        ne_labels: List[int] = [
            (
                NE_TAGS.index("O")
                if morpheme_global_index in word_feature_example.global_index2feature_set.keys()
                else IGNORE_INDEX
            )
            for morpheme_global_index in range(self.max_seq_length)
        ]
        for named_entity in word_feature_example.named_entities:
            category = named_entity.category.value
            for i, morpheme in enumerate(named_entity.morphemes):
                bi = "B" if i == 0 else "I"
                assert ne_labels[morpheme.global_index] == NE_TAGS.index("O"), f"nested NE found in {example.doc_id}"
                ne_index = NE_TAGS.index(f"{bi}-{category}")
                ne_labels[morpheme.global_index] = ne_index

        # ---------- base phrase feature tagging ----------
        base_phrase_feature_example = example.base_phrase_feature_example
        base_phrase_feature_labels = [[IGNORE_INDEX] * len(BASE_PHRASE_FEATURES) for _ in range(self.max_seq_length)]
        for (
            head_morpheme_global_index,
            feature_set,
        ) in base_phrase_feature_example.head_morpheme_global_index2feature_set.items():
            for i, base_phrase_feature in enumerate(BASE_PHRASE_FEATURES):
                base_phrase_feature_labels[head_morpheme_global_index][i] = int(base_phrase_feature in feature_set)

        # ---------- dependency parsing ----------
        dependency_example = example.dependency_example
        root_index = self.special_to_index["[ROOT]"]
        dependency_labels: List[int] = [IGNORE_INDEX for _ in range(self.max_seq_length)]
        for morpheme_global_index, dependency in dependency_example.global_index2dependency.items():
            dependency_labels[morpheme_global_index] = dependency if dependency >= 0 else root_index
        # True/False = keep/mask
        dependency_mask = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        for morpheme_global_index, candidates in dependency_example.global_index2candidates.items():
            for candidate_index in candidates + [root_index]:
                dependency_mask[morpheme_global_index][candidate_index] = True
        dependency_type_labels: List[int] = [IGNORE_INDEX for _ in range(self.max_seq_length)]
        for morpheme_global_index, dependency_type in dependency_example.global_index2dependency_type.items():
            dependency_type_index = DEPENDENCY_TYPES.index(dependency_type)
            dependency_type_labels[morpheme_global_index] = dependency_type_index

        # ---------- cohesion analysis ----------
        cohesion_example = example.cohesion_example
        cohesion_target: List[List[List[int]]] = []  # (task, src, tgt)
        candidates_list: List[List[List[int]]] = []  # (task, src, tgt)
        if CohesionTask.PAS_ANALYSIS in self.cohesion_tasks:
            task = CohesionTask.PAS_ANALYSIS
            annotation = cohesion_example.annotations[task]
            phrases = cohesion_example.phrases[task]
            for case in self.pas_cases:
                assert type(annotation.arguments_set) == List[Dict[str, List[str]]]
                arguments_set = [arguments[case] for arguments in annotation.arguments_set]
                ret = self._convert_annotation_to_feature(arguments_set, phrases)
                cohesion_target.append(ret[0])
                candidates_list.append(ret[1])
        for task in (CohesionTask.BRIDGING, CohesionTask.COREFERENCE):
            if task not in self.cohesion_tasks:
                continue
            annotation = cohesion_example.annotations[task]
            assert type(annotation.arguments_set) == List[List[str]]
            arguments_set = annotation.arguments_set
            phrases = cohesion_example.phrases[task]
            ret = self._convert_annotation_to_feature(arguments_set, phrases)
            cohesion_target.append(ret[0])
            candidates_list.append(ret[1])
        # True/False = keep/mask
        cohesion_mask = [
            [[(x in cs) for x in range(self.max_seq_length)] for cs in candidates] for candidates in candidates_list
        ]

        # ---------- discourse parsing ----------
        discourse_example = example.discourse_example
        discourse_labels = [[IGNORE_INDEX for _ in range(self.max_seq_length)] for _ in range(self.max_seq_length)]
        for modifier_morpheme_global_index, relations in enumerate(discourse_example.discourse_relations):
            for head_morpheme_global_index, relation in enumerate(relations):
                if relation in DISCOURSE_RELATIONS:
                    discourse_index = DISCOURSE_RELATIONS.index(relation)
                    discourse_labels[modifier_morpheme_global_index][head_morpheme_global_index] = discourse_index

        return {
            "example_ids": torch.tensor(example.example_id, dtype=torch.long),
            "input_ids": torch.tensor(merged_encoding.ids, dtype=torch.long),
            "attention_mask": torch.tensor(merged_encoding.attention_mask, dtype=torch.long),
            "target_mask": torch.tensor(target_mask, dtype=torch.long),
            "subword_map": torch.tensor(self._get_subword_map(merged_encoding), dtype=torch.bool),
            "reading_labels": torch.tensor(reading_labels, dtype=torch.long),
            "reading_subword_map": torch.tensor(
                self._get_subword_map(merged_encoding, include_special_tokens=False), dtype=torch.bool
            ),
            "pos_labels": morpheme_attribute_labels[:, 0],
            "subpos_labels": morpheme_attribute_labels[:, 1],
            "conjtype_labels": morpheme_attribute_labels[:, 2],
            "conjform_labels": morpheme_attribute_labels[:, 3],
            "word_feature_labels": torch.tensor(word_feature_labels, dtype=torch.long),
            "ne_labels": torch.tensor(ne_labels, dtype=torch.long),
            "base_phrase_feature_labels": torch.tensor(base_phrase_feature_labels, dtype=torch.long),
            "dependency_labels": torch.tensor(dependency_labels, dtype=torch.long),
            "dependency_mask": torch.tensor(dependency_mask, dtype=torch.bool),
            "dependency_type_labels": torch.tensor(dependency_type_labels, dtype=torch.long),
            "cohesion_target": torch.tensor(cohesion_target, dtype=torch.int),
            "cohesion_mask": torch.tensor(cohesion_mask, dtype=torch.bool),
            "discourse_labels": torch.tensor(discourse_labels, dtype=torch.long),
        }

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
        return [index for special, index in self.special_to_index.items() if special != "[ROOT]"]

    @property
    def cohesion_rel_types(self) -> List[str]:
        return [t for task in self.cohesion_tasks for t in self.cohesion_task_to_rel_types[task]]

    @property
    def num_special_tokens(self) -> int:
        return len(self.special_tokens)

    @property
    def special_indices(self) -> List[int]:
        return list(self.special_to_index.values())

    def _convert_annotation_to_feature(
        self,
        arguments_set: List[List[str]],  # phrase level
        phrases: List[Phrase],
    ) -> Tuple[List[List[int]], List[List[int]]]:
        scores_list: List[List[int]] = [[0] * self.max_seq_length for _ in range(self.max_seq_length)]
        candidates_list: List[List[int]] = [[] for _ in range(self.max_seq_length)]
        for phrase in phrases:
            arguments: List[str] = arguments_set[phrase.dtid]
            candidates: List[int] = phrase.candidates  # phrase level
            for mrph in phrase.children:
                scores: List[int] = [0] * self.max_seq_length
                if mrph.is_target:
                    for arg_string in arguments:
                        # arg_string: 著者, 8%C, 15%O, 2, [NULL], ...
                        if arg_string[-2:] in ("%C", "%N", "%O"):
                            # PAS only
                            # flag = arg_string[-1]
                            arg_string = arg_string[:-2]
                        if arg_string in self.special_to_index:
                            word_index = self.special_to_index[arg_string]
                        else:
                            word_index = phrases[int(arg_string)].dmid
                        scores[word_index] = 1

                word_level_candidates: List[int] = []
                for candidate in candidates:
                    word_level_candidates.append(phrases[candidate].dmid)
                word_level_candidates += self.cohesion_special_indices

                # use the head subword as the representative of the source word
                scores_list[mrph.dmid] = scores
                candidates_list[mrph.dmid] = word_level_candidates

        return scores_list, candidates_list  # word level
