import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union
from unicodedata import normalize

import torch
from rhoknp import Document, Sentence
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from kwja.datamodule.datasets.base_dataset import BaseDataset
from kwja.datamodule.examples.char_feature import CharFeatureExample
from kwja.utils.constants import (
    IGNORE_INDEX,
    IGNORE_WORD_NORM_OP_TAG,
    TRANSLATION_TABLE,
    WORD_NORM_OP_TAGS,
    WORD_SEGMENTATION_TAGS,
)
from kwja.utils.progress_bar import track
from kwja.utils.word_normalization import SentenceDenormalizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CharExampleSet:
    example_id: int
    doc_id: str
    text: str  # space-delimited word sequence
    encoding: BatchEncoding
    char_feature_example: CharFeatureExample


class CharDataset(BaseDataset):
    def __init__(
        self,
        path: str,
        document_split_stride: int,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 512,
        denormalize_probability: float = 0.0,
    ) -> None:
        self.path = Path(path)
        super().__init__(self.path, tokenizer, max_seq_length, document_split_stride)
        self.denormalizer: SentenceDenormalizer = SentenceDenormalizer()
        self.denormalize_probability: float = denormalize_probability
        self.examples: List[CharExampleSet] = self._load_examples(self.documents)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.encode(self.examples[index])

    def _load_examples(self, documents: List[Document]) -> List[CharExampleSet]:
        examples = []
        example_id = 0
        for document in track(documents, description="Loading examples"):
            for sentence in document.sentences:
                self.denormalizer.denormalize(sentence, self.denormalize_probability)
            encoding: BatchEncoding = self.tokenizer(
                document.text,
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=False,
                max_length=self.max_seq_length,
            )
            if len(encoding.input_ids) > self.max_seq_length:
                logger.warning(f"Length of sub document is too long: {document.text}")
                continue

            char_feature_example = CharFeatureExample()
            char_feature_example.load(document)

            examples.append(
                CharExampleSet(
                    example_id=example_id,
                    doc_id=document.doc_id,
                    text=document.text,
                    encoding=encoding,
                    char_feature_example=char_feature_example,
                )
            )
            example_id += 1
        if len(examples) == 0:
            logger.error(
                "No examples to process. "
                f"Make sure there exist any documents in {self.path} and they are not too long."
            )
        return examples

    def encode(self, example: CharExampleSet) -> Dict[str, torch.Tensor]:
        char_feature_example = example.char_feature_example

        word_segmentation_labels: List[int] = [IGNORE_INDEX for _ in range(self.max_seq_length)]
        for char_index, word_segmentation_tag in char_feature_example.index2word_segmentation_tag.items():
            # 先頭の[CLS]をIGNORE_INDEXにするため+1
            word_segmentation_labels[char_index + 1] = WORD_SEGMENTATION_TAGS.index(word_segmentation_tag)

        word_norm_op_labels: List[int] = [IGNORE_INDEX for _ in range(self.max_seq_length)]
        for char_index, word_norm_op_tag in char_feature_example.index2word_norm_op_tag.items():
            # 先頭の[CLS]をIGNORE_INDEXにするため+1
            if word_norm_op_tag == IGNORE_WORD_NORM_OP_TAG:
                word_norm_op_labels[char_index + 1] = IGNORE_INDEX
            else:
                word_norm_op_labels[char_index + 1] = WORD_NORM_OP_TAGS.index(word_norm_op_tag)

        return {
            "example_ids": torch.tensor(example.example_id, dtype=torch.long),
            "input_ids": torch.tensor(example.encoding.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(example.encoding.attention_mask, dtype=torch.long),
            "word_segmentation_labels": torch.tensor(word_segmentation_labels, dtype=torch.long),
            "word_norm_op_labels": torch.tensor(word_norm_op_labels, dtype=torch.long),
        }

    def _normalize_text(self, document: Document) -> Document:
        for morpheme in document.morphemes:
            normalized = normalize("NFKC", morpheme.text).translate(TRANSLATION_TABLE)
            if normalized != morpheme.text:
                logger.warning(f"apply normalization ({morpheme.text} -> {normalized})")
                morpheme.text = normalized
                morpheme.lemma = normalize("NFKC", morpheme.lemma).translate(TRANSLATION_TABLE)
        return document

    def _get_tokenized_len(self, document_or_sentence: Union[Document, Sentence]) -> int:
        return len(self.tokenizer.tokenize(document_or_sentence.text))
