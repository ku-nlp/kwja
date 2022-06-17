import torch
from rhoknp import Document
from transformers import BatchEncoding

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
