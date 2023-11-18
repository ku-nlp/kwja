import logging
from typing import Dict, List, Optional

from omegaconf import ListConfig
from rhoknp import Document, RegexSenter
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from kwja.datamodule.datasets.base import BaseDataset, FullAnnotatedDocumentLoaderMixin
from kwja.datamodule.datasets.char import CharModuleFeatures
from kwja.datamodule.datasets.utils import add_doc_ids, add_sent_ids, create_documents_from_raw_texts
from kwja.datamodule.examples import CharInferenceExample
from kwja.utils.logging_util import track

logger = logging.getLogger(__name__)


class CharInferenceDataset(BaseDataset[CharInferenceExample, CharModuleFeatures], FullAnnotatedDocumentLoaderMixin):
    def __init__(
        self,
        texts: ListConfig,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        doc_id_prefix: Optional[str] = None,
        **_,
    ) -> None:
        super().__init__(tokenizer, max_seq_length)
        documents = create_documents_from_raw_texts(texts)
        add_doc_ids(documents, doc_id_prefix)
        documents = self._add_tentative_sentence_boundary(documents)
        super(BaseDataset, self).__init__(documents, tokenizer, max_seq_length, -1)  # document_split_stride must be -1
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
            sent_segmentation_labels=[],
            word_segmentation_labels=[],
            word_norm_op_labels=[],
        )

    @staticmethod
    def _add_tentative_sentence_boundary(documents: List[Document]) -> List[Document]:
        senter = RegexSenter()
        documents = [senter.apply_to_document(doc) if doc.is_senter_required() else doc for doc in documents]
        add_sent_ids(documents)
        return documents
