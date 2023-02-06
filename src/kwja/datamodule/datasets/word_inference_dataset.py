import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import torch
from omegaconf import ListConfig
from rhoknp import Document, Morpheme, Sentence
from rhoknp.cohesion import ExophoraReferent
from rhoknp.utils.reader import chunk_by_document
from tokenizers import Encoding
from transformers.utils import PaddingStrategy

from kwja.datamodule.datasets.base_dataset import BaseDataset
from kwja.datamodule.examples import CohesionTask
from kwja.datamodule.extractors import BridgingExtractor, CoreferenceExtractor, PasExtractor
from kwja.utils.constants import SPLIT_INTO_WORDS_MODEL_NAMES
from kwja.utils.progress_bar import track
from kwja.utils.sub_document import extract_target_sentences

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WordInferenceExample:
    example_id: int
    doc_id: str
    encoding: Encoding


class WordInferenceDataset(BaseDataset):
    def __init__(
        self,
        document_split_stride: int,
        pas_cases: ListConfig,
        bar_rels: ListConfig,
        exophora_referents: ListConfig,
        cohesion_tasks: ListConfig,
        restrict_cohesion_target: bool,
        special_tokens: ListConfig,
        juman_file: Optional[Path] = None,
        knp_file: Optional[Path] = None,
        model_name_or_path: str = "nlp-waseda/roberta-base-japanese",
        tokenizer_kwargs: Optional[dict] = None,
        max_seq_length: int = 512,
        **_,  # accept reading_resource_path
    ) -> None:
        if juman_file is not None:
            with juman_file.open(mode="r") as f:
                documents = [
                    Document.from_jumanpp(c) for c in track(chunk_by_document(f), description="Loading documents")
                ]
        elif knp_file is not None:
            with knp_file.open(mode="r") as f:
                documents = [Document.from_knp(c) for c in track(chunk_by_document(f), description="Loading documents")]
        else:
            # do_predict_after_train
            documents = []

        if model_name_or_path in SPLIT_INTO_WORDS_MODEL_NAMES:
            self.tokenizer_input_format: Literal["words", "text"] = "words"
        else:
            self.tokenizer_input_format = "text"

        super().__init__(
            documents,
            model_name_or_path,
            max_seq_length,
            document_split_stride,
            tokenizer_kwargs=tokenizer_kwargs,
        )

        # ---------- cohesion analysis ----------
        self.pas_cases: List[str] = list(pas_cases)
        self.bar_rels: List[str] = list(bar_rels)
        self.exophora_referents = [ExophoraReferent(s) for s in exophora_referents]
        self.cohesion_tasks = [CohesionTask(t) for t in cohesion_tasks]
        self.cohesion_task_to_rel_types = {
            CohesionTask.PAS_ANALYSIS: self.pas_cases,
            CohesionTask.BRIDGING: self.bar_rels,
            CohesionTask.COREFERENCE: ["="],
        }
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

        self.examples: List[WordInferenceExample] = self._load_examples(self.documents)

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

    def _load_examples(self, documents: List[Document]) -> List[WordInferenceExample]:
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
            num_tokenized_morphemes = len({word_id for word_id in encoding.word_ids if word_id is not None})
            if len(document.morphemes) != num_tokenized_morphemes:
                logger.warning(f"Document length and tokenized length mismatch: {document.text}")
                continue

            examples.append(
                WordInferenceExample(
                    example_id=example_id,
                    doc_id=document.doc_id,
                    encoding=encoding,
                )
            )
            example_id += 1
        if len(examples) == 0:
            logger.error("No examples to process. Make sure any texts are given and they are not too long.")
        return examples

    def encode(self, example: WordInferenceExample) -> Dict[str, torch.Tensor]:
        document = self.doc_id2document[example.doc_id]

        target_mask = [0 for _ in range(self.max_seq_length)]
        for sentence in extract_target_sentences(document):
            for morpheme in sentence.morphemes:
                target_mask[morpheme.global_index] = 1

        # ---------- dependency parsing ----------
        # True/False = keep/mask
        dependency_mask = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        for sentence in document.sentences:
            morpheme_global_indices = [morpheme.global_index for morpheme in sentence.morphemes]
            start, stop = min(morpheme_global_indices), max(morpheme_global_indices) + 1
            for i in range(start, stop):
                for j in range(start, stop):
                    if i != j:
                        dependency_mask[i][j] = True
                dependency_mask[i][self.special_to_index["[ROOT]"]] = True

        # ---------- cohesion analysis ----------
        candidates_list: List[List[List[int]]] = []  # (task, src, tgt)
        morphemes = document.morphemes
        if CohesionTask.PAS_ANALYSIS in self.cohesion_tasks:
            candidates: List[List[int]] = [[] for _ in range(self.max_seq_length)]
            for i, source_morpheme in enumerate(morphemes):
                candidates[i] = self._get_candidate_morpheme_global_indices(morphemes, source_morpheme)
                candidates[i] += self.cohesion_special_indices
            candidates_list.extend([candidates] * len(self.pas_cases))
        if CohesionTask.BRIDGING in self.cohesion_tasks:
            candidates = [[] for _ in range(self.max_seq_length)]
            for i, source_morpheme in enumerate(morphemes):
                candidates[i] = self._get_candidate_morpheme_global_indices(morphemes, source_morpheme)
                candidates[i] += self.cohesion_special_indices
            candidates_list.append(candidates)
        if CohesionTask.COREFERENCE in self.cohesion_tasks:
            candidates = [[] for _ in range(self.max_seq_length)]
            for i, source_morpheme in enumerate(morphemes):
                candidates[i] = self._get_candidate_morpheme_global_indices(
                    morphemes, source_morpheme, coreference=True
                )
                candidates[i] += self.cohesion_special_indices
            candidates_list.append(candidates)
        # True/False = keep/mask
        cohesion_mask = [
            [[(x in cs) for x in range(self.max_seq_length)] for cs in candidates] for candidates in candidates_list
        ]

        merged_encoding: Encoding = Encoding.merge([example.encoding, self.special_encoding])

        return {
            "example_ids": torch.tensor(example.example_id, dtype=torch.long),
            "input_ids": torch.tensor(merged_encoding.ids, dtype=torch.long),
            "attention_mask": torch.tensor(merged_encoding.attention_mask, dtype=torch.long),
            "target_mask": torch.tensor(target_mask, dtype=torch.long),
            "subword_map": torch.tensor(self._get_subword_map(merged_encoding), dtype=torch.bool),
            "reading_subword_map": torch.tensor(
                self._get_subword_map(merged_encoding, include_special_tokens=False), dtype=torch.bool
            ),
            "dependency_mask": torch.tensor(dependency_mask, dtype=torch.bool),
            "cohesion_mask": torch.tensor(cohesion_mask, dtype=torch.bool),
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
        return [t for ts in self.cohesion_task_to_rel_types.values() for t in ts]

    @property
    def num_special_tokens(self) -> int:
        return len(self.special_tokens)

    @property
    def special_indices(self) -> List[int]:
        return list(self.special_to_index.values())

    @staticmethod
    def _get_candidate_morpheme_global_indices(
        morphemes: List[Morpheme],
        source_morpheme: Morpheme,
        coreference: bool = False,
    ) -> List[int]:
        candidates: List[int] = []
        for target_morpheme in morphemes:
            if target_morpheme.global_index < source_morpheme.global_index:
                candidates.append(target_morpheme.global_index)
            elif coreference is False:
                if (
                    target_morpheme.global_index > source_morpheme.global_index
                    and target_morpheme.sentence.sid == source_morpheme.sentence.sid
                ):
                    candidates.append(target_morpheme.global_index)
        return candidates
