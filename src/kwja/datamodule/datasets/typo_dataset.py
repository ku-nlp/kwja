import json
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from kwja.utils.constants import TYPO_DUMMY_TOKEN, TYPO_OPN2TOKEN


class TypoDataset(Dataset):
    def __init__(
        self,
        path: str,
        extended_vocab_path: str,
        model_name_or_path: str = "ku-nlp/roberta-base-japanese-char-wwm",
        max_seq_length: int = 512,
        tokenizer_kwargs: dict = None,
    ) -> None:
        self.path = Path(path)
        assert self.path.is_dir()

        self.documents = self.load_documents(self.path)
        assert len(self) != 0

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **(tokenizer_kwargs or {}),
        )
        assert self.tokenizer.pad_token_id is not None
        self.pad_token_id: int = self.tokenizer.pad_token_id
        assert self.tokenizer.unk_token_id is not None
        self.unk_token_id: int = self.tokenizer.unk_token_id
        self.max_seq_length: int = max_seq_length

        self.opn2id: dict[str, int] = self.get_opn2id(path=Path(extended_vocab_path))

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.encode(self.documents[index])

    @staticmethod
    def load_documents(path: Path) -> list[dict[str, Union[str, list[str]]]]:
        documents: list[dict[str, Union[str, list[str]]]] = []
        for file_path in sorted(path.glob("**/*.jsonl")):
            with file_path.open(mode="r", encoding="utf-8") as f:
                for line in f:
                    documents.append(json.loads(line))
        return documents

    def get_opn2id(self, path: Path) -> dict[str, int]:
        opn2id = self.tokenizer.get_vocab()
        with path.open(mode="r") as f:
            for line in f:
                opn2id[str(line.strip())] = len(opn2id)
        return opn2id

    def encode(self, document: dict[str, Union[str, list[str]]]) -> dict[str, torch.Tensor]:
        if isinstance(document["pre_text"], list):
            raise ValueError('document["pre_text"] must be string')
        encoding: BatchEncoding = self.tokenizer(
            document["pre_text"] + TYPO_DUMMY_TOKEN,
            truncation=True,
            padding=PaddingStrategy.MAX_LENGTH,
            max_length=self.max_seq_length,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        kdr_labels: list[int] = []
        for opn in document["kdrs"][:-1]:
            if opn in TYPO_OPN2TOKEN:
                kdr_label = self.opn2id[TYPO_OPN2TOKEN[opn]]
            else:
                kdr_label = self.opn2id.get(opn.removeprefix("R:"), self.unk_token_id)
            kdr_labels.append(kdr_label)
        kdr_labels.append(self.pad_token_id)
        kdr_labels = [self.pad_token_id] + kdr_labels[: self.max_seq_length - 2] + [self.pad_token_id]
        kdr_labels = kdr_labels + [self.pad_token_id] * (self.max_seq_length - len(kdr_labels))

        ins_labels: list[int] = []
        for opn in document["inss"]:
            if opn in TYPO_OPN2TOKEN:
                ins_label = self.opn2id[TYPO_OPN2TOKEN[opn]]
            else:
                ins_label = self.opn2id.get(opn.removeprefix("I:"), self.unk_token_id)
            ins_labels.append(ins_label)
        ins_labels = [self.pad_token_id] + ins_labels[: self.max_seq_length - 2] + [self.pad_token_id]
        ins_labels = ins_labels + [self.pad_token_id] * (self.max_seq_length - len(ins_labels))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "kdr_labels": torch.tensor(kdr_labels, dtype=torch.long),
            "ins_labels": torch.tensor(ins_labels, dtype=torch.long),
            "texts": document["pre_text"],
        }
