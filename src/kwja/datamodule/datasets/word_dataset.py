import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
from omegaconf import ListConfig
from rhoknp import Document, Sentence
from rhoknp.cohesion import ExophoraReferent
from scipy.special import softmax
from tokenizers import Encoding
from tqdm import tqdm
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
    CONJFORM_TYPES,
    CONJTYPE_TYPES,
    DEPENDENCY_TYPE2INDEX,
    DISCOURSE_RELATIONS,
    IGNORE_INDEX,
    NE_TAGS,
    POS_TYPES,
    SUBPOS_TYPES,
    WORD_FEATURES,
)
from kwja.utils.kanjidic import KanjiDic
from kwja.utils.reading import ReadingAligner, get_reading2id
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
        pas_cases: ListConfig,
        bar_rels: ListConfig,
        exophora_referents: ListConfig,
        cohesion_tasks: ListConfig,
        special_tokens: ListConfig,
        restrict_cohesion_target: bool,
        reading_resource_path: str,
        document_split_stride: int,
        model_name_or_path: str = "nlp-waseda/roberta-base-japanese",
        max_seq_length: int = 512,
        tokenizer_kwargs: dict = None,
    ) -> None:
        self.path = Path(path)
        super().__init__(
            self.path,
            document_split_stride,
            model_name_or_path,
            max_seq_length,
            tokenizer_kwargs or {},
        )
        self.special_tokens: list[str] = list(special_tokens)
        self.exophora_referents = [ExophoraReferent(s) for s in exophora_referents]
        self.cohesion_tasks: list[CohesionTask] = [CohesionTask(t) for t in cohesion_tasks]
        self.pas_cases: list[str] = list(pas_cases)
        self.bar_rels: list[str] = list(bar_rels)
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
        self.special_to_index: dict[str, int] = {
            token: self.max_seq_length - len(self.special_tokens) + i for i, token in enumerate(self.special_tokens)
        }
        self.index_to_special: dict[int, str] = {v: k for k, v in self.special_to_index.items()}
        self.reading_resource_path = Path(reading_resource_path)
        self.reading2id = get_reading2id(str(self.reading_resource_path / "vocab.txt"))
        self.reading_aligner = ReadingAligner(self.tokenizer, KanjiDic(str(self.reading_resource_path / "kanjidic")))
        self.examples: list[WordExampleSet] = self._load_examples(self.documents)
        self.special_encoding: Encoding = self.tokenizer(
            self.special_tokens,
            is_split_into_words=True,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=False,
            add_special_tokens=False,
        ).encodings[0]

    @property
    def special_indices(self) -> list[int]:
        return list(self.special_to_index.values())

    @property
    def cohesion_special_indices(self) -> list[int]:
        return [index for special, index in self.special_to_index.items() if special != "[ROOT]"]

    @property
    def num_special_tokens(self) -> int:
        return len(self.special_tokens)

    @property
    def cohesion_rel_types(self) -> list[str]:
        return [t for task in self.cohesion_tasks for t in self.cohesion_task_to_rel_types[task]]

    def _load_examples(self, documents: list[Document]) -> list[WordExampleSet]:
        examples = []
        idx = 0
        for document in tqdm(documents, dynamic_ncols=True):
            encoding: Encoding = self.tokenizer(
                [morpheme.text for morpheme in document.morphemes],
                is_split_into_words=True,
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=False,
                max_length=self.max_seq_length - self.num_special_tokens,
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
                    document_disc = Document.from_knp(path.read_text())
                    if document == document_disc:
                        discourse_example.load(document_disc)
                except AssertionError:
                    logger.warning(f"{path} is not a valid KNP file")

            examples.append(
                WordExampleSet(
                    example_id=idx,
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
            idx += 1

        if len(examples) == 0:
            logger.error(
                "No examples to process. "
                f"Make sure there exist any documents in {self.path} and they are not too long."
            )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.encode(self.examples[index])

    def encode(self, example: WordExampleSet) -> dict[str, torch.Tensor]:
        merged_encoding: Encoding = Encoding.merge([example.encoding, self.special_encoding])

        document = self.doc_id2document[example.doc_id]
        target_mask = [False for _ in range(self.max_seq_length)]
        for sentence in extract_target_sentences(document):
            for morpheme in sentence.morphemes:
                target_mask[morpheme.global_index] = True

        # reading prediction
        reading_example = example.reading_example
        reading_ids = [IGNORE_INDEX] * self.max_seq_length
        if reading_example.readings:
            non_special_token_indexes = [
                token_index
                for token_index, word_id in enumerate(merged_encoding.word_ids)
                if word_id is not None and token_index not in self.index_to_special
            ]
            for index, non_special_token_index in enumerate(non_special_token_indexes):
                reading = reading_example.readings[index]
                decoded_token = self.tokenizer.decode(merged_encoding.ids[non_special_token_index])
                if reading == decoded_token:
                    reading_ids[non_special_token_index] = self.reading2id["[ID]"]
                else:
                    reading_ids[non_special_token_index] = self.reading2id.get(reading, self.reading2id["[UNK]"])

        # NOTE: hereafter, indices are given at the word level

        # morpheme type tagging
        word_feature_example = example.word_feature_example
        morpheme_type_set = (POS_TYPES, SUBPOS_TYPES, CONJTYPE_TYPES, CONJFORM_TYPES)
        morpheme_types = [[IGNORE_INDEX] * len(morpheme_type_set) for _ in range(self.max_seq_length)]
        for morpheme_index, mrph_types in word_feature_example.types.items():
            for i, (mrph_type, all_types) in enumerate(zip(mrph_types, morpheme_type_set)):
                if mrph_type in all_types:
                    morpheme_types[morpheme_index][i] = all_types.index(mrph_type)

        ne_tags: list[int] = [
            (NE_TAGS.index("O") if morpheme_index in word_feature_example.features.keys() else IGNORE_INDEX)
            for morpheme_index in range(self.max_seq_length)
        ]
        for named_entity in word_feature_example.named_entities:
            category = named_entity.category.value
            for i, morpheme in enumerate(named_entity.morphemes):
                bi = "B" if i == 0 else "I"
                assert ne_tags[morpheme.global_index] == NE_TAGS.index("O"), f"nested NE found in {example.doc_id}"
                ne_tags[morpheme.global_index] = NE_TAGS.index(f"{bi}-{category}")

        # word feature tagging
        word_features = [[IGNORE_INDEX] * len(WORD_FEATURES) for _ in range(self.max_seq_length)]
        for morpheme_index, feature_set in word_feature_example.features.items():
            for i, word_feature in enumerate(WORD_FEATURES):
                word_features[morpheme_index][i] = int(word_feature in feature_set)

        # base phrase feature tagging
        base_phrase_feature_example = example.base_phrase_feature_example
        base_phrase_features = [[IGNORE_INDEX] * len(BASE_PHRASE_FEATURES) for _ in range(self.max_seq_length)]
        for head_index, feature_set in zip(base_phrase_feature_example.heads, base_phrase_feature_example.features):
            for i, base_phrase_feature in enumerate(BASE_PHRASE_FEATURES):
                base_phrase_features[head_index][i] = int(base_phrase_feature in feature_set)

        # dependency parsing
        dependency_example = example.dependency_example
        dependencies: list[int] = [IGNORE_INDEX for _ in range(self.max_seq_length)]
        for morpheme_index, dependency in dependency_example.dependencies.items():
            dependencies[morpheme_index] = dependency if dependency != -1 else self.special_to_index["[ROOT]"]

        # False -> mask, True -> keep
        dependency_mask = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        for morpheme_index, candidates in dependency_example.candidates.items():
            for candidate_index in candidates + [self.special_to_index["[ROOT]"]]:
                dependency_mask[morpheme_index][candidate_index] = True

        dependency_types: list[int] = [IGNORE_INDEX for _ in range(self.max_seq_length)]
        for morpheme_index, dependency_type in dependency_example.dependency_types.items():
            dependency_types[morpheme_index] = DEPENDENCY_TYPE2INDEX[dependency_type]

        # PAS analysis & coreference resolution
        cohesion_example = example.cohesion_example
        cohesion_target: list[list[list[int]]] = []  # (task, src, tgt)
        candidates_set: list[list[list[int]]] = []  # (task, src, tgt)
        if CohesionTask.PAS_ANALYSIS in self.cohesion_tasks:
            task = CohesionTask.PAS_ANALYSIS
            annotation = cohesion_example.annotations[task]
            phrases = cohesion_example.phrases[task]
            for case in self.pas_cases:
                arguments_set = [arguments[case] for arguments in annotation.arguments_set]
                ret = self._convert_annotation_to_feature(arguments_set, phrases)
                cohesion_target.append(ret[0])
                candidates_set.append(ret[1])
        for task in (CohesionTask.BRIDGING, CohesionTask.COREFERENCE):
            if task not in self.cohesion_tasks:
                continue
            annotation = cohesion_example.annotations[task].arguments_set
            phrases = cohesion_example.phrases[task]
            ret = self._convert_annotation_to_feature(annotation, phrases)
            cohesion_target.append(ret[0])
            candidates_set.append(ret[1])
        cohesion_mask = [
            [[(x in cands) for x in range(self.max_seq_length)] for cands in candidates]
            for candidates in candidates_set
        ]  # False -> mask, True -> keep

        discourse_example = example.discourse_example
        discourse_relations = [[IGNORE_INDEX for _ in range(self.max_seq_length)] for _ in range(self.max_seq_length)]
        for global_morpheme_index_i, relations in enumerate(discourse_example.discourse_relations):
            for global_morpheme_index_j, relation in enumerate(relations):
                if relation in DISCOURSE_RELATIONS:
                    relation_index = DISCOURSE_RELATIONS.index(relation)
                    discourse_relations[global_morpheme_index_i][global_morpheme_index_j] = relation_index

        return {
            "example_ids": torch.tensor(example.example_id, dtype=torch.long),
            "input_ids": torch.tensor(merged_encoding.ids, dtype=torch.long),
            "attention_mask": torch.tensor(merged_encoding.attention_mask, dtype=torch.long),
            "target_mask": torch.tensor(target_mask, dtype=torch.bool),
            "subword_map": torch.tensor(self._gen_subword_map(merged_encoding), dtype=torch.bool),
            "reading_subword_map": torch.tensor(
                self._gen_subword_map(merged_encoding, include_additional_words=False), dtype=torch.bool
            ),
            "mrph_types": torch.tensor(morpheme_types, dtype=torch.long),
            "reading_ids": torch.tensor(reading_ids, dtype=torch.long),
            "ne_tags": torch.tensor(ne_tags, dtype=torch.long),
            "word_features": torch.tensor(word_features, dtype=torch.long),
            "base_phrase_features": torch.tensor(base_phrase_features, dtype=torch.long),
            "dependencies": torch.tensor(dependencies, dtype=torch.long),
            "intra_mask": torch.tensor(dependency_mask, dtype=torch.bool),
            "dependency_types": torch.tensor(dependency_types, dtype=torch.long),
            "discourse_relations": torch.tensor(discourse_relations, dtype=torch.long),
            "cohesion_target": torch.tensor(cohesion_target, dtype=torch.int),
            "cohesion_mask": torch.tensor(cohesion_mask, dtype=torch.bool),
            "tokens": " ".join(self.tokenizer.decode(id_) for id_ in merged_encoding.ids),
        }

    def dump_prediction(
        self,
        result: list[list[list[float]]],  # word level
        example: CohesionExample,
    ) -> list[list[list[float]]]:  # (phrase, rel, 0 or phrase+special)
        """1 example 中に存在する基本句それぞれに対してシステム予測のリストを返す．"""
        ret: list[list[list[float]]] = [[] for _ in next(iter(example.phrases.values()))]
        task_idx = 0
        if CohesionTask.PAS_ANALYSIS in self.cohesion_tasks:
            for _ in self.pas_cases:
                for i, p in enumerate(
                    self._word2bp_level(result[task_idx], example.phrases[CohesionTask.PAS_ANALYSIS])
                ):
                    ret[i].append(p)
                task_idx += 1
        if CohesionTask.BRIDGING in self.cohesion_tasks:
            for i, p in enumerate(self._word2bp_level(result[task_idx], example.phrases[CohesionTask.BRIDGING])):
                ret[i].append(p)
            task_idx += 1
        if CohesionTask.COREFERENCE in self.cohesion_tasks:
            for i, p in enumerate(self._word2bp_level(result[task_idx], example.phrases[CohesionTask.COREFERENCE])):
                ret[i].append(p)
            task_idx += 1
        return ret

    def _word2bp_level(
        self,
        word_level_output: list[list[float]],
        phrases: list[Phrase],
    ) -> list[list[float]]:  # (bp, 0 or bp+special)
        ret: list[list[float]] = [[] for _ in phrases]
        for anaphor in filter(lambda p: p.is_target, phrases):
            word_level_scores: list[float] = word_level_output[anaphor.dmid]
            phrase_level_scores: list[float] = []
            for tgt_bp in phrases:
                if tgt_bp.dtid not in anaphor.candidates:
                    phrase_level_scores.append(-1)  # pad -1 for non-candidate phrases
                    continue
                phrase_level_scores.append(word_level_scores[tgt_bp.dmid])
            phrase_level_scores += [word_level_scores[idx] for idx in self.special_indices]
            assert len(phrase_level_scores) == len(phrases) + len(self.special_to_index)
            ret[anaphor.dtid] = softmax(phrase_level_scores).tolist()
        return ret

    def _gen_subword_map(self, encoding: Encoding, include_additional_words: bool = True) -> list[list[bool]]:
        subword_map = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        for token_id, word_id in enumerate(encoding.word_ids):
            if word_id is None or token_id in self.special_indices:
                continue
            subword_map[word_id][token_id] = True
        if include_additional_words:
            for special_index in self.special_indices:
                subword_map[special_index][special_index] = True
        return subword_map

    def _convert_annotation_to_feature(
        self,
        annotation: list[list[str]],  # phrase level
        phrases: list[Phrase],
    ) -> tuple[list[list[int]], list[list[int]]]:
        scores_set: list[list[int]] = [[0] * self.max_seq_length for _ in range(self.max_seq_length)]
        candidates_set: list[list[int]] = [[] for _ in range(self.max_seq_length)]
        for phrase in phrases:
            arguments: list[str] = annotation[phrase.dtid]
            candidates: list[int] = phrase.candidates  # phrase level
            for mrph in phrase.children:
                scores: list[int] = [0] * self.max_seq_length
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

                word_level_candidates: list[int] = []
                for candidate in candidates:
                    word_level_candidates.append(phrases[candidate].dmid)
                word_level_candidates += self.cohesion_special_indices

                # use the head subword as the representative of the source word
                scores_set[mrph.dmid] = scores
                candidates_set[mrph.dmid] = word_level_candidates

        return scores_set, candidates_set  # word level

    def _get_tokenized_len(self, source: Union[Document, Sentence]) -> int:
        return len(
            self.tokenizer([m.text for m in source.morphemes], add_special_tokens=False, is_split_into_words=True)[
                "input_ids"
            ]
        )
