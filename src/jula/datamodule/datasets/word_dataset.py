import torch
from transformers import BatchEncoding

from jula.datamodule.datasets.base_dataset import BaseDataset


class WordDataset(BaseDataset):
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        document = self.documents[index]

        # TODO: deal with the case that the document is too long
        encoding: BatchEncoding = self.tokenizer(
            " ".join(morpheme.text for morpheme in document.morphemes),
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length,
        )

        return {
            "input_ids": torch.tensor(
                encoding["input_ids"],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                encoding["attention_mask"],
                dtype=torch.long,
            ),
        }
