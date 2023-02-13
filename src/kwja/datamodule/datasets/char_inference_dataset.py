import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union
from unicodedata import normalize

import torch
from omegaconf import ListConfig
from rhoknp import Document, RegexSenter, Sentence
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

import kwja
from kwja.datamodule.datasets.base_dataset import BaseDataset
from kwja.utils.constants import TRANSLATION_TABLE
from kwja.utils.progress_bar import track

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CharInferenceExample:
    example_id: int
    doc_id: str
    encoding: BatchEncoding


class CharInferenceDataset(BaseDataset):
    def __init__(
        self,
        texts: ListConfig,
        tokenizer: PreTrainedTokenizerBase,
        document_split_stride: int,
        max_seq_length: int,
        doc_id_prefix: Optional[str] = None,
        **_,
    ) -> None:
        documents = self._build_documents_from_texts(list(texts), doc_id_prefix)
        super().__init__(documents, tokenizer, max_seq_length, document_split_stride)
        self.examples: List[CharInferenceExample] = self._load_examples(self.documents)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.encode(self.examples[index])

    def _load_examples(self, documents: List[Document]) -> List[CharInferenceExample]:
        examples = []
        example_id = 0
        for document in track(documents, description="Loading examples"):
            encoding: BatchEncoding = self.tokenizer(
                document.text,
                truncation=False,
                padding=PaddingStrategy.MAX_LENGTH,
                max_length=self.max_seq_length,
            )
            if len(encoding.input_ids) > self.max_seq_length:
                logger.warning(f"Length of sub document is too long: {document.text}")
                continue

            examples.append(
                CharInferenceExample(
                    example_id=example_id,
                    doc_id=document.doc_id,
                    encoding=encoding,
                )
            )
            example_id += 1
        if len(examples) == 0:
            logger.error("No examples to process. Make sure any texts are given and they are not too long.")
        return examples

    @staticmethod
    def encode(example: CharInferenceExample) -> Dict[str, torch.Tensor]:
        return {
            "example_ids": torch.tensor(example.example_id, dtype=torch.long),
            "input_ids": torch.tensor(example.encoding.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(example.encoding.attention_mask, dtype=torch.long),
        }

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
        for document_index, document in enumerate(documents):
            document.doc_id = f"{doc_id_prefix}-{document_index:0{doc_id_width}}"
            for sentence_index, sentence in enumerate(document.sentences):
                sentence.sid = f"{document.doc_id}-{sentence_index:0{sent_id_width}}"
                sentence.misc_comment = f"kwja:{kwja.__version__}"
        return documents

    def _normalize(self, document):
        for sentence in document.sentences:
            normalized = normalize("NFKC", sentence.text).translate(TRANSLATION_TABLE)
            if normalized != sentence.text:
                logger.warning(f"apply normalization ({sentence.text} -> {normalized})")
                sentence.text = normalized
        document.text = "".join(sentence.text for sentence in document.sentences)
        return document

    def _get_tokenized_len(self, document_or_sentence: Union[Document, Sentence]) -> int:
        return len(self.tokenizer.tokenize(document_or_sentence.text))
