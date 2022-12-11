import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union
from unicodedata import normalize

import torch
from omegaconf import ListConfig
from rhoknp import Document, RegexSenter, Sentence
from transformers import BatchEncoding
from transformers.utils import PaddingStrategy

import kwja
from kwja.datamodule.datasets.base_inference_dataset import BaseInferenceDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CharInferenceExample:
    example_id: int
    doc_id: str
    encoding: BatchEncoding


class CharInferenceDataset(BaseInferenceDataset):
    def __init__(
        self,
        texts: ListConfig,
        document_split_stride: int,
        model_name_or_path: str = "ku-nlp/roberta-base-japanese-char-wwm",
        tokenizer_kwargs: Optional[dict] = None,
        max_seq_length: int = 512,
        doc_id_prefix: Optional[str] = None,
        **_,
    ) -> None:
        super().__init__(
            list(texts),
            document_split_stride,
            model_name_or_path,
            max_seq_length,
            tokenizer_kwargs or {},
            doc_id_prefix=doc_id_prefix,
        )
        self.examples: List[CharInferenceExample] = self._load_examples(self.documents)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.encode(self.examples[index])

    def _load_examples(self, documents: List[Document]) -> List[CharInferenceExample]:
        examples = []
        idx = 0
        for document in documents:
            encoding: BatchEncoding = self.tokenizer(
                document.text,
                truncation=False,
                padding=PaddingStrategy.MAX_LENGTH,
                max_length=self.max_seq_length,
            )
            if len(encoding["input_ids"]) > self.max_seq_length:
                continue
            if len(document.text) != self._get_tokenized_len(document):
                logger.warning(f"Document length and tokenized length mismatch: {document.text}")
                continue

            examples.append(
                CharInferenceExample(
                    example_id=idx,
                    doc_id=document.doc_id,
                    encoding=encoding,
                )
            )
            idx += 1
        if len(examples) == 0:
            logger.error("No examples to process. Make sure any texts are given and they are not too long.")
        return examples

    @staticmethod
    def encode(example: CharInferenceExample) -> Dict[str, torch.Tensor]:
        input_ids = example.encoding["input_ids"]
        attention_mask = example.encoding["attention_mask"]
        return {
            "example_ids": torch.tensor(example.example_id, dtype=torch.long),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def _create_documents_from_texts(self, texts: List[str], doc_id_prefix: Optional[str]) -> List[Document]:
        senter = RegexSenter()
        # split text into sentences
        documents: List[Document] = [senter.apply_to_document(text) for text in texts]
        if doc_id_prefix is None:
            doc_id_prefix = datetime.now().strftime("%Y%m%d%H%M")
        doc_id_width = len(str(len(documents)))
        sent_id_width = max((len(str(len(doc.sentences))) for doc in documents), default=0)
        for doc_idx, document in enumerate(documents):
            document.doc_id = f"{doc_id_prefix}-{doc_idx:0{doc_id_width}}"
            for sent_idx, sentence in enumerate(document.sentences):
                sentence.sid = f"{document.doc_id}-{sent_idx:0{sent_id_width}}"
                sentence.misc_comment = f"kwja:{kwja.__version__}"
                sentence.text = self._normalize(sentence.text)
            document.text = "".join(sentence.text for sentence in document.sentences)
        return documents

    def _get_tokenized_len(self, source: Union[Document, Sentence]) -> int:
        return len(self.tokenizer.tokenize(source.text))

    def _normalize(self, text):
        normalized = normalize("NFKC", text).replace(" ", "␣").replace('"', "＂")
        return normalized
