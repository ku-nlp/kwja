import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from rhoknp import Document
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from kwja.datamodule.datasets.base import BaseDataset, FullAnnotatedDocumentLoaderMixin
from kwja.datamodule.examples import SenterExample
from kwja.utils.constants import IGNORE_INDEX, SENT_SEGMENTATION_TAGS
from kwja.utils.progress_bar import track

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SenterModuleFeatures:
    example_ids: int
    input_ids: List[int]
    attention_mask: List[int]
    sent_segmentation_labels: List[int]


class SenterDataset(BaseDataset[SenterExample, SenterModuleFeatures], FullAnnotatedDocumentLoaderMixin):
    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 512,
        document_split_stride: int = -1,
    ) -> None:
        super(SenterDataset, self).__init__(tokenizer, max_seq_length)
        self.path = Path(path)
        super(BaseDataset, self).__init__(self.path, tokenizer, max_seq_length, document_split_stride)
        self.examples: List[SenterExample] = self._load_examples(self.documents)

    def _load_examples(self, documents: List[Document]) -> List[SenterExample]:
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

            example = SenterExample(example_id, encoding)
            example.load_document(document)
            examples.append(example)
            example_id += 1

        if len(examples) == 0:
            logger.error(
                "No examples to process. "
                f"Make sure there exist any documents in {self.path} and they are not too long."
            )
        return examples

    def encode(self, example: SenterExample) -> SenterModuleFeatures:
        sent_segmentation_labels: List[int] = [IGNORE_INDEX for _ in range(self.max_seq_length)]
        for char_global_index, sent_segmentation_tag in example.char_global_index2sent_segmentation_tag.items():
            # 先頭の[CLS]をIGNORE_INDEXにするため+1
            sent_segmentation_labels[char_global_index + 1] = SENT_SEGMENTATION_TAGS.index(sent_segmentation_tag)
        return SenterModuleFeatures(
            example_ids=example.example_id,
            input_ids=example.encoding.input_ids,
            attention_mask=example.encoding.attention_mask,
            sent_segmentation_labels=sent_segmentation_labels,
        )

    def _postprocess_document(self, document: Document) -> Document:
        for i in reversed(range(len(document.sentences))):
            if "括弧位置" in document.sentences[i].comment:
                del document.sentences[i]
        return document