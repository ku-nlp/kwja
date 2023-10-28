import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from unicodedata import normalize

from rhoknp import Document
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from kwja.datamodule.datasets.base import BaseDataset, FullAnnotatedDocumentLoaderMixin
from kwja.datamodule.examples import CharExample
from kwja.utils.constants import (
    IGNORE_INDEX,
    IGNORE_WORD_NORM_OP_TAG,
    SENT_SEGMENTATION_TAGS,
    TRANSLATION_TABLE,
    WORD_NORM_OP_TAGS,
    WORD_SEGMENTATION_TAGS,
)
from kwja.utils.logging_util import track
from kwja.utils.word_normalization import SentenceDenormalizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CharModuleFeatures:
    example_ids: int
    input_ids: List[int]
    attention_mask: List[int]
    sent_segmentation_labels: List[int]
    word_segmentation_labels: List[int]
    word_norm_op_labels: List[int]


class CharDataset(BaseDataset[CharExample, CharModuleFeatures], FullAnnotatedDocumentLoaderMixin):
    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        denormalize_probability: float,
    ) -> None:
        super().__init__(tokenizer, max_seq_length)
        self.path = Path(path)
        self.denormalizer = SentenceDenormalizer()
        is_training = self.path.parts[-1] == "train" or (
            self.path.parts[-2] == "kyoto_ed" and self.path.parts[-1] == "all"
        )
        self.denormalize_probability: float = denormalize_probability if is_training else 0.0
        super(BaseDataset, self).__init__(self.path, tokenizer, max_seq_length, -1)  # document_split_stride must be -1
        self.examples: List[CharExample] = self._load_examples(self.doc_id2document)
        if is_training is True:
            del self.doc_id2document  # for saving memory

    def _load_examples(self, doc_id2document: Dict[str, Document]) -> List[CharExample]:
        examples = []
        example_id = 0
        for document in track(doc_id2document.values(), description="Loading examples"):
            encoding: BatchEncoding = self.tokenizer(
                document.text,
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=False,
                max_length=self.max_seq_length,
            )
            if len(encoding.input_ids) > self.max_seq_length:
                logger.warning(f"Length of sub document is too long: {document.text}")
                continue

            example = CharExample(example_id, encoding)
            example.load_document(document)
            examples.append(example)
            example_id += 1
        if len(examples) == 0:
            logger.error(
                "No examples to process. "
                f"Make sure there exist any documents in {self.path} and they are not too long."
            )
        return examples

    def encode(self, example: CharExample) -> CharModuleFeatures:
        sent_segmentation_labels: List[int] = [IGNORE_INDEX for _ in range(self.max_seq_length)]
        for char_global_index, sent_segmentation_tag in example.char_global_index2sent_segmentation_tag.items():
            # 先頭の[CLS]をIGNORE_INDEXにするため+1
            sent_segmentation_labels[char_global_index + 1] = SENT_SEGMENTATION_TAGS.index(sent_segmentation_tag)

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
            sent_segmentation_labels=sent_segmentation_labels,
            word_segmentation_labels=word_segmentation_labels,
            word_norm_op_labels=word_norm_op_labels,
        )

    def _postprocess_document(self, document: Document) -> Document:
        for i in reversed(range(len(document.sentences))):
            sentence = document.sentences[i]
            if "括弧位置" in sentence.comment:
                del document.sentences[i]
            else:
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
