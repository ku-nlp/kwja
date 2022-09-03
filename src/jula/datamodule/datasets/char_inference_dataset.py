from datetime import datetime
from typing import Optional, Union

import torch
from omegaconf import ListConfig
from rhoknp import Document, RegexSenter, Sentence
from transformers import BatchEncoding
from transformers.utils import PaddingStrategy

import jula
from jula.datamodule.datasets.base_dataset import BaseDataset


class CharInferenceDataset(BaseDataset):
    def __init__(
        self,
        texts: ListConfig,
        document_split_stride: int,
        model_name_or_path: str = "cl-tohoku/bert-base-japanese-char",
        max_seq_length: int = 512,
        tokenizer_kwargs: dict = None,
        doc_id_prefix: Optional[str] = None,
    ) -> None:
        documents = self._create_documents_from_texts(list(texts), doc_id_prefix)
        super().__init__(documents, document_split_stride, model_name_or_path, max_seq_length, tokenizer_kwargs or {})

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.encode(self.documents[index])

    def encode(self, document: Document) -> dict[str, torch.Tensor]:
        encoding: BatchEncoding = self.tokenizer(
            document.text,
            truncation=True,
            padding=PaddingStrategy.MAX_LENGTH,
            max_length=self.max_seq_length,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    @staticmethod
    def _create_documents_from_texts(texts: list[str], doc_id_prefix: Optional[str]) -> list[Document]:
        senter = RegexSenter()
        # split text into sentences
        documents: list[Document] = [senter.apply_to_document(text) for text in texts]
        if doc_id_prefix is None:
            doc_id_prefix = datetime.now().strftime("%Y%m%d%H%M")
        doc_id_width = len(str(len(documents)))
        sent_id_width = max(len(str(len(doc.sentences))) for doc in documents)
        for doc_idx, document in enumerate(documents):
            document.doc_id = f"{doc_id_prefix}-{doc_idx:0{doc_id_width}}"
            for sent_idx, sentence in enumerate(document.sentences):
                sentence.sid = f"{document.doc_id}-{sent_idx:0{sent_id_width}}"
                sentence.misc_comment = f"jula:{jula.__version__}"
        return documents

    def _get_tokenized_len(self, source: Union[Document, Sentence]) -> int:
        return len(self.tokenizer(source.text, add_special_tokens=False)["input_ids"])
