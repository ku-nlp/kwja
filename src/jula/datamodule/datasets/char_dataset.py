import json
from pathlib import Path
from typing import Union

import torch
from rhoknp import Document
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase

from jula.datamodule.datasets.base_dataset import BaseDataset
from jula.utils.utils import SEG_LABEL2INDEX


class CharDataset(BaseDataset):
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.encode(self.documents[index])

    def encode(self, document: Document) -> dict[str, torch.Tensor]:
        encoding: BatchEncoding = self.tokenizer(
            document.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        seg_labels: list[int] = []
        for morpheme in document.morphemes:
            seg_labels.extend(
                [SEG_LABEL2INDEX["B"]]
                + [SEG_LABEL2INDEX["I"]] * (len(morpheme.text) - 1)
            )
        seg_labels = (
            [SEG_LABEL2INDEX["PAD"]]
            + seg_labels[: self.max_seq_length - 2]
            + [SEG_LABEL2INDEX["PAD"]]
        )
        seg_labels = seg_labels + [SEG_LABEL2INDEX["PAD"]] * (
            self.max_seq_length - len(seg_labels)
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "seg_labels": torch.tensor(seg_labels, dtype=torch.long),
        }


class CharTypoDataset(Dataset):
    def __init__(
        self,
        path: str,
        insert_vocab_path: str,
        model_name_or_path: str = "cl-tohoku/bert-base-japanese-char",
        max_seq_length: int = 512,
        tokenizer_kwargs: dict = None,
    ) -> None:
        self.path = Path(path)
        assert self.path.is_dir()

        self.documents = self.load_documents(self.path)
        assert len(self) != 0

        tokenizer_kwargs = tokenizer_kwargs or {}
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **tokenizer_kwargs,
        )
        self.max_seq_length = max_seq_length

        self.insert_vocab = self.make_insert_vocab(path=Path(insert_vocab_path))

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

    def make_insert_vocab(self, path: Path) -> dict[str, int]:
        vocab = self.tokenizer.get_vocab()
        with path.open(mode="r") as f:
            for line in f:
                vocab[str(line.strip())] = len(vocab)
        return vocab

    def encode(
        self, document: list[dict[str, Union[str, list[str]]]]
    ) -> dict[str, torch.Tensor]:
        encoding: BatchEncoding = self.tokenizer(
            document["pre_text"] + "<0x00>",
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        kdr_labels: list[int] = []
        for ops in document["kdrs"][:-1]:
            if ops == "K":
                kdr_label = self.insert_vocab["<0x01>"]
            elif ops == "D":
                kdr_label = self.insert_vocab["<0x02>"]
            else:
                kdr_label = self.insert_vocab.get(
                    ops.removeprefix("R:"), self.tokenizer.unk_token_id
                )
            kdr_labels.append(kdr_label)
        kdr_labels.append(self.tokenizer.pad_token_id)

        kdr_labels = (
            [self.tokenizer.pad_token_id]
            + kdr_labels[: self.max_seq_length - 2]
            + [self.tokenizer.pad_token_id]
        )
        kdr_labels = kdr_labels + [self.tokenizer.pad_token_id] * (
            self.max_seq_length - len(kdr_labels)
        )

        ins_labels: list[int] = []
        for ops in document["inss"]:
            if ops == "_":
                ins_label = self.insert_vocab["<0x03>"]
            else:
                ins_label = self.insert_vocab.get(
                    ops.removeprefix("I:"), self.tokenizer.unk_token_id
                )
            ins_labels.append(ins_label)
        ins_labels = (
            [self.tokenizer.pad_token_id]
            + ins_labels[: self.max_seq_length - 2]
            + [self.tokenizer.pad_token_id]
        )
        ins_labels = ins_labels + [self.tokenizer.pad_token_id] * (
            self.max_seq_length - len(ins_labels)
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "kdr_labels": torch.tensor(kdr_labels, dtype=torch.long),
            "ins_labels": torch.tensor(ins_labels, dtype=torch.long),
        }
