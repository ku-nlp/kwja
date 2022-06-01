import torch
from rhoknp import Document
from transformers import BatchEncoding

from jula.datamodule.datasets.base_dataset import BaseDataset
from jula.utils.features import BASE_PHRASE_FEATURES


class WordDataset(BaseDataset):
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.encode(self.documents[index])

    def encode(self, document: Document) -> dict[str, torch.Tensor]:
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

        # hereafter, indices are given at the word level
        base_phrase_features = [
            [0] * len(BASE_PHRASE_FEATURES) for _ in range(self.max_seq_length)
        ]
        for base_phrase in document.base_phrases:
            for i, base_phrase_feature in enumerate(BASE_PHRASE_FEATURES):
                if ":" in base_phrase_feature:
                    key, value = base_phrase_feature.split(":")
                else:
                    key, value = base_phrase_feature, ""
                if base_phrase.features.get(key, False) in (value, True):
                    base_phrase_features[base_phrase.head.global_index][i] = 1
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
            "base_phrase_features": torch.tensor(
                base_phrase_features,
                dtype=torch.float,
            ),
        }
