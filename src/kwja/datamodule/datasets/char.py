import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union
from unicodedata import normalize

from rhoknp import Document, Sentence
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from kwja.datamodule.datasets.base import BaseDataset
from kwja.datamodule.examples import CharExample
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
class CharModuleFeatures:
    example_ids: int
    input_ids: List[int]
    attention_mask: List[int]
    word_segmentation_labels: List[int]
    word_norm_op_labels: List[int]


class CharDataset(BaseDataset[CharModuleFeatures]):
    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        denormalize_probability: float,
        document_split_stride: int = -1,
    ) -> None:
        self.path = Path(path)
        self.denormalizer: SentenceDenormalizer = SentenceDenormalizer()
        self.denormalize_probability: float = denormalize_probability
        super().__init__(self.path, tokenizer, max_seq_length, document_split_stride)
        self.examples: List[CharExample] = self._load_examples(self.documents)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> CharModuleFeatures:
        return self.encode(self.examples[index])

    def _load_examples(self, documents: List[Document]) -> List[CharExample]:
        examples = []
        example_id = 0
        for document in track(documents, description="Loading examples"):
            encoding: BatchEncoding = self.tokenizer(
                document.text,
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=False,
                max_length=self.max_seq_length,
            )
            if len(encoding.input_ids) > self.max_seq_length:
                logger.warning(f"Length of sub document is too long: {document.text}")
                continue

            char_example = CharExample(example_id, encoding)
            char_example.load_document(document)

            examples.append(char_example)
            example_id += 1
        if len(examples) == 0:
            logger.error(
                "No examples to process. "
                f"Make sure there exist any documents in {self.path} and they are not too long."
            )
        return examples

    def encode(self, example: CharExample) -> CharModuleFeatures:
        word_segmentation_labels: List[int] = [IGNORE_INDEX for _ in range(self.max_seq_length)]
        for char_global_index, word_segmentation_tag in example.char_global_index2word_segmentation_tag.items():
            # 先頭の[CLS]をIGNORE_INDEXにするため+1
            word_segmentation_labels[char_global_index + 1] = WORD_SEGMENTATION_TAGS.index(word_segmentation_tag)

        word_norm_op_labels: List[int] = [IGNORE_INDEX for _ in range(self.max_seq_length)]
        for char_global_index, word_norm_op_tag in example.char_global_index2word_norm_op_tag.items():
            # 先頭の[CLS]をIGNORE_INDEXにするため+1
            if word_norm_op_tag == IGNORE_WORD_NORM_OP_TAG:
                word_norm_op_labels[char_global_index + 1] = IGNORE_INDEX
            else:
                word_norm_op_labels[char_global_index + 1] = WORD_NORM_OP_TAGS.index(word_norm_op_tag)

        return CharModuleFeatures(
            example_ids=example.example_id,
            input_ids=example.encoding.input_ids,
            attention_mask=example.encoding.attention_mask,
            word_segmentation_labels=word_segmentation_labels,
            word_norm_op_labels=word_norm_op_labels,
        )

    def _normalize_text(self, document: Document) -> Document:
        for sentence in document.sentences:
            # e.g. です -> でーす
            self.denormalizer.denormalize(sentence, self.denormalize_probability)
            for morpheme in sentence.morphemes:
                normalized = normalize("NFKC", morpheme.text).translate(TRANSLATION_TABLE)
                if normalized != morpheme.text:
                    logger.warning(f"apply normalization ({morpheme.text} -> {normalized})")
                    morpheme.text = normalized
                    morpheme.lemma = normalize("NFKC", morpheme.lemma).translate(TRANSLATION_TABLE)
        # propagate updates of morpheme.text to sentence.text and document.text
        return document.reparse()

    def _get_tokenized_len(self, document_or_sentence: Union[Document, Sentence]) -> int:
        return len(self.tokenizer.tokenize(document_or_sentence.text))