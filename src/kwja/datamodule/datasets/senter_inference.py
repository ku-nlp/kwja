import logging
from datetime import datetime
from typing import Dict, List, Optional

from omegaconf import ListConfig
from rhoknp import Document, RegexSenter
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

import kwja
from kwja.datamodule.datasets.base import BaseDataset, FullAnnotatedDocumentLoaderMixin
from kwja.datamodule.datasets.senter import SenterModuleFeatures
from kwja.datamodule.examples import SenterInferenceExample
from kwja.utils.logging_util import track

logger = logging.getLogger(__name__)


class SenterInferenceDataset(
    BaseDataset[SenterInferenceExample, SenterModuleFeatures], FullAnnotatedDocumentLoaderMixin
):
    def __init__(
        self,
        texts: ListConfig,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        document_split_stride: int = -1,
        doc_id_prefix: Optional[str] = None,
        **_,
    ) -> None:
        super(SenterInferenceDataset, self).__init__(tokenizer, max_seq_length)
        documents = self._build_documents_from_texts(list(texts), doc_id_prefix)
        super(BaseDataset, self).__init__(documents, tokenizer, max_seq_length, document_split_stride)
        self.examples: List[SenterInferenceExample] = self._load_examples(self.doc_id2document)

    def _load_examples(self, doc_id2document: Dict[str, Document]) -> List[SenterInferenceExample]:
        examples = []
        example_id = 0
        for document in track(doc_id2document.values(), description="Loading examples"):
            encoding: BatchEncoding = self.tokenizer(
                document.text,
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=False,
                max_length=self.max_seq_length,
            )
            if len(encoding["input_ids"]) > self.max_seq_length:
                logger.warning(f"Length of sub document is too long: {document.text}")
                continue

            example = SenterInferenceExample(example_id, encoding, document.doc_id)
            examples.append(example)
            example_id += 1
        if len(examples) == 0:
            logger.error("No examples to process. Make sure any texts are given and they are not too long.")
        return examples

    def encode(self, example: SenterInferenceExample) -> SenterModuleFeatures:
        return SenterModuleFeatures(
            example_ids=example.example_id,
            input_ids=example.encoding.input_ids,
            attention_mask=example.encoding.attention_mask,
            sent_segmentation_labels=[],
        )

    @staticmethod
    def _build_documents_from_texts(texts: List[str], doc_id_prefix: Optional[str]) -> List[Document]:
        senter = RegexSenter()
        # split text into sentences
        documents: List[Document] = [
            senter.apply_to_document(text) for text in track(texts, description="Loading documents")
        ]
        if doc_id_prefix is None:
            doc_id_prefix = datetime.now().strftime("%Y%m%d%H%M")
        doc_id_width = len(str(len(documents)))
        sent_id_width = max((len(str(len(doc.sentences))) for doc in documents), default=0)
        for doc_idx, document in enumerate(documents):
            document.doc_id = f"{doc_id_prefix}-{doc_idx:0{doc_id_width}}"
            for sent_idx, sentence in enumerate(document.sentences):
                sentence.sid = f"{document.doc_id}-{sent_idx:0{sent_id_width}}"
                sentence.misc_comment = f"kwja:{kwja.__version__}"
        return documents
