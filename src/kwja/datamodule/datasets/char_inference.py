import logging
from pathlib import Path
from typing import Dict, List, Optional

from rhoknp import Document
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from kwja.datamodule.datasets.base import BaseDataset, FullAnnotatedDocumentLoaderMixin
from kwja.datamodule.datasets.char import CharModuleFeatures
from kwja.datamodule.examples import CharInferenceExample
from kwja.utils.logging_util import track
from kwja.utils.reader import chunk_by_document_for_line_by_line_text

logger = logging.getLogger(__name__)


class CharInferenceDataset(BaseDataset[CharInferenceExample, CharModuleFeatures], FullAnnotatedDocumentLoaderMixin):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        document_split_stride: int = -1,
        senter_file: Optional[Path] = None,
        **_,
    ) -> None:
        super(CharInferenceDataset, self).__init__(tokenizer, max_seq_length)
        if senter_file is not None:
            with senter_file.open() as f:
                documents = [
                    Document.from_line_by_line_text(c)
                    for c in track(chunk_by_document_for_line_by_line_text(f), description="Loading documents")
                ]
        else:
            documents = []
        super(BaseDataset, self).__init__(documents, tokenizer, max_seq_length, document_split_stride)
        self.examples: List[CharInferenceExample] = self._load_examples(self.doc_id2document)

    def _load_examples(self, doc_id2document: Dict[str, Document]) -> List[CharInferenceExample]:
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

            examples.append(CharInferenceExample(example_id=example_id, encoding=encoding, doc_id=document.doc_id))
            example_id += 1
        if len(examples) == 0:
            logger.error("No examples to process. Make sure any texts are given and they are not too long.")
        return examples

    def encode(self, example: CharInferenceExample) -> CharModuleFeatures:
        return CharModuleFeatures(
            example_ids=example.example_id,
            input_ids=example.encoding.input_ids,
            attention_mask=example.encoding.attention_mask,
            word_segmentation_labels=[],
            word_norm_op_labels=[],
        )
