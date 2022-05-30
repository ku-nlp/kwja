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

        subword_map = [
            [False] * self.max_seq_length for _ in range(self.max_seq_length)
        ]
        for token_id, word_id in enumerate(encoding.word_ids()):
            if word_id is not None:
                subword_map[word_id][token_id] = True

        return {
            "input_ids": torch.tensor(
                encoding["input_ids"],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                encoding["attention_mask"],
                dtype=torch.long,
            ),
            "subword_map": torch.tensor(
                subword_map,
                dtype=torch.bool,
            ),
        }
