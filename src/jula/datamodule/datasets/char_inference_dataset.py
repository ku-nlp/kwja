from datetime import datetime
from typing import Optional

import torch
from rhoknp import Document, RegexSenter
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


class CharInferenceDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        model_name_or_path: str = "nlp-waseda/roberta-base-japanese",
        max_seq_length: int = 512,
        tokenizer_kwargs: dict = None,
        doc_id_prefix: Optional[str] = None,
        **_,
    ) -> None:
        senter = RegexSenter()
        # split text into sentences
        self.documents: list[Document] = [senter.apply_to_document(text) for text in texts]
        if doc_id_prefix is None:
            doc_id_prefix = datetime.now().strftime("%Y%m%d%H%M")
        doc_id_width = len(str(len(self.documents)))
        sent_id_width = max(len(str(len(doc.sentences))) for doc in self.documents)
        for doc_idx, document in enumerate(self.documents):
            document.doc_id = f"{doc_id_prefix}{doc_idx:0{doc_id_width}}"
            for sent_idx, sentence in enumerate(document.sentences):
                sentence.sid = f"{document.doc_id}-{sent_idx:0{sent_id_width}}"
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **(tokenizer_kwargs or {}),
        )
        self.max_seq_length = max_seq_length

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
