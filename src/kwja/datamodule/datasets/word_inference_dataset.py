import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import torch
from omegaconf import ListConfig
from rhoknp import Document, Morpheme, Sentence
from rhoknp.cohesion import ExophoraReferent
from rhoknp.units.morpheme import MorphemeAttributes
from tokenizers import Encoding
from transformers.utils import PaddingStrategy

import kwja
from kwja.datamodule.datasets.base_dataset import BaseDataset
from kwja.datamodule.examples import CohesionTask
from kwja.datamodule.extractors import BridgingExtractor, CoreferenceExtractor, PasExtractor
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
        texts: ListConfig,
        pas_cases: ListConfig,
        bar_rels: ListConfig,
        exophora_referents: ListConfig,
        cohesion_tasks: ListConfig,
        special_tokens: ListConfig,
        restrict_cohesion_target: bool,
        document_split_stride: int,
        model_name_or_path: str = "nlp-waseda/roberta-base-japanese",
        max_seq_length: int = 512,
        tokenizer_kwargs: dict = None,
        doc_id_prefix: Optional[str] = None,
        knp_file: Optional[Path] = None,
        **_,  # accept reading_resource_path
    ) -> None:
        if knp_file is None:
            documents = self._create_documents_from_texts(list(texts), doc_id_prefix)
        else:
            documents = [Document.from_knp(knp_file.read_text())]
        super().__init__(documents, document_split_stride, model_name_or_path, max_seq_length, tokenizer_kwargs or {})

        self.exophora_referents = [ExophoraReferent(s) for s in exophora_referents]
        self.special_tokens: list[str] = list(special_tokens)
        self.special_to_index: dict[str, int] = {
            token: self.max_seq_length - len(self.special_tokens) + i for i, token in enumerate(self.special_tokens)
        }
        self.index_to_special: dict[int, str] = {v: k for k, v in self.special_to_index.items()}
        self.cohesion_tasks = [CohesionTask(t) for t in cohesion_tasks]
        self.pas_cases: list[str] = list(pas_cases)
        self.bar_rels: list[str] = list(bar_rels)
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
        self.examples: list[WordInferenceExample] = self._load_examples(self.documents)
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
        return [t for ts in self.cohesion_task_to_rel_types.values() for t in ts]

    @staticmethod
    def _create_documents_from_texts(texts: list[str], doc_id_prefix: Optional[str]) -> list[Document]:
        did2sentences = defaultdict(list)
        if doc_id_prefix is None:
            doc_id_prefix = datetime.now().strftime("%Y%m%d%H%M")
        doc_id, sid = f"{doc_id_prefix}-0", f"{doc_id_prefix}-0-0"
        for text in texts:
            if text.startswith("#"):
                sentence = Sentence.from_raw_text(text)
                doc_id, sid = sentence.doc_id, sentence.sid
            else:
                sentence = Sentence()
                sentence.doc_id, sentence.sid = doc_id, sid
                sentence.misc_comment = f"kwja:{kwja.__version__}"
                morphemes = []
                for word in text.split(" "):
                    attributes = MorphemeAttributes(
                        reading="null",
                        lemma="null",
                        pos="null",
                        pos_id=0,
                        subpos="null",
                        subpos_id=0,
                        conjtype="null",
                        conjtype_id=0,
                        conjform="null",
                        conjform_id=0,
                    )
                    morphemes.append(Morpheme(word, attributes))
                sentence.morphemes = morphemes
                did2sentences[doc_id].append(sentence)
        return [Document.from_sentences(sentences) for sentences in did2sentences.values()]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.encode(self.examples[index])

    def _load_examples(self, documents: list[Document]) -> list[WordInferenceExample]:
        examples = []
        idx = 0
        for document in documents:
            encoding: Encoding = self.tokenizer(
                [morpheme.text for morpheme in document.morphemes],
                is_split_into_words=True,
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=False,
                max_length=self.max_seq_length - self.num_special_tokens,
            ).encodings[0]
            if len(encoding.ids) > self.max_seq_length - self.num_special_tokens:
                continue
            num_tokenized_morphemes = len({word_id for word_id in encoding.word_ids if word_id is not None})
            if len(document.morphemes) != num_tokenized_morphemes:
                logger.warning(f"Document length and tokenized length mismatch: {document.text}")
                continue

            examples.append(
                WordInferenceExample(
                    example_id=idx,
                    doc_id=document.doc_id,
                    encoding=encoding,
                )
            )
            idx += 1

        if len(examples) == 0:
            logger.error("No examples to process. Make sure any texts are given and they are not too long.")
        return examples

    def encode(self, example: WordInferenceExample) -> dict[str, torch.Tensor]:
        document = self.doc_id2document[example.doc_id]

        target_mask = [False for _ in range(self.max_seq_length)]
        for sentence in extract_target_sentences(document):
            for morpheme in sentence.morphemes:
                target_mask[morpheme.global_index] = True

        dependency_mask = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        for sentence in document.sentences:
            morpheme_global_indices = [morpheme.global_index for morpheme in sentence.morphemes]
            start, stop = min(morpheme_global_indices), max(morpheme_global_indices) + 1
            for i in range(start, stop):
                for j in range(start, stop):
                    if i != j:
                        dependency_mask[i][j] = True
                dependency_mask[i][self.special_to_index["[ROOT]"]] = True

        cohesion_mask = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        num_morphemes = len(document.morphemes)
        for i in range(num_morphemes):
            for j in range(num_morphemes):
                if i != j:
                    cohesion_mask[i][j] = True
            for special_index in self.cohesion_special_indices:
                cohesion_mask[i][special_index] = True

        merged_encoding: Encoding = Encoding.merge([example.encoding, self.special_encoding])

        return {
            "example_ids": torch.tensor(example.example_id, dtype=torch.long),
            "input_ids": torch.tensor(merged_encoding.ids, dtype=torch.long),
            "attention_mask": torch.tensor(merged_encoding.attention_mask, dtype=torch.long),
            "target_mask": torch.tensor(target_mask, dtype=torch.bool),
            "subword_map": torch.tensor(self._gen_subword_map(merged_encoding), dtype=torch.bool),
            "reading_subword_map": torch.tensor(
                self._gen_subword_map(merged_encoding, include_additional_words=False), dtype=torch.bool
            ),
            "intra_mask": torch.tensor(dependency_mask, dtype=torch.bool),
            "cohesion_mask": torch.tensor(cohesion_mask, dtype=torch.bool)
            .unsqueeze(0)
            .expand(len(self.cohesion_rel_types), self.max_seq_length, self.max_seq_length),
            "tokens": " ".join(self.tokenizer.decode(id_) for id_ in merged_encoding.ids),
        }

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

    def _get_tokenized_len(self, source: Union[Document, Sentence]) -> int:
        return len(
            self.tokenizer([m.text for m in source.morphemes], add_special_tokens=False, is_split_into_words=True)[
                "input_ids"
            ]
        )
