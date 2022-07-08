import pickle

import dartsclone
import torch
from rhoknp import Document
from transformers import BatchEncoding
from transformers.utils import PaddingStrategy

from jula.datamodule.datasets.base_dataset import BaseDataset
from jula.utils.utils import ENE_TYPE_BIES, IGNORE_INDEX, SEG_TYPES


class CharDataset(BaseDataset):
    def __init__(
        self,
        wiki_ene_dic_path: str,
        max_ene_num: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.max_ene_num = max_ene_num
        self.darts = dartsclone.DoubleArray()
        self.darts.open(f"{wiki_ene_dic_path}/wiki.da")
        self.values: list[list[str]] = pickle.load(
            open(f"{wiki_ene_dic_path}/wiki_values.pkl", "rb")
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        document = self.documents[index]
        return {
            "example_ids": torch.tensor(index, dtype=torch.long),
            **self.encode(document),
        }

    def get_ene_ids(self, text: str) -> list[list[int]]:
        pos2ene_ids: dict[int, list[int]] = {
            pos: [] for pos in range(self.max_seq_length)
        }
        for char_pos in range(0, len(text) - 1):
            subtext: bytes = text[char_pos:].encode("utf-8")
            for match_idx, match_len in self.darts.common_prefix_search(subtext):
                match_word_len: int = len(subtext[0:match_len].decode("utf-8"))
                for i in range(match_word_len):
                    pos_in_match_word: int = char_pos + i
                    for ene_type in self.values[match_idx]:
                        if i == 0:
                            bie = "B"
                        elif i == match_word_len - 1:
                            bie = "E"
                        else:
                            bie = "I"
                        ene_id: int = ENE_TYPE_BIES.index(f"{bie}-{ene_type}")
                        if ene_id not in pos2ene_ids[pos_in_match_word]:
                            pos2ene_ids[pos_in_match_word].append(ene_id)
        ene_ids: list[list[int]] = [
            [ENE_TYPE_BIES.index("PAD")] * self.max_seq_length
            for _ in range(self.max_ene_num)
        ]  # (max_ene_num, max_seq_length)
        for pos in pos2ene_ids.keys():
            for idx, ene_id in enumerate(pos2ene_ids[pos][: self.max_ene_num]):
                ene_ids[idx][pos] = ene_id

        return ene_ids

    def encode(self, document: Document) -> dict[str, torch.Tensor]:
        encoding: BatchEncoding = self.tokenizer(
            document.text,
            truncation=True,
            padding=PaddingStrategy.MAX_LENGTH,
            max_length=self.max_seq_length,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        seg_labels: list[int] = []
        for morpheme in document.morphemes:
            seg_labels.extend(
                [SEG_TYPES.index("B")]
                + [SEG_TYPES.index("I")] * (len(morpheme.text) - 1)
            )
        seg_labels = (
            [IGNORE_INDEX] + seg_labels[: self.max_seq_length - 2] + [IGNORE_INDEX]
        )
        seg_labels = seg_labels + [IGNORE_INDEX] * (
            self.max_seq_length - len(seg_labels)
        )

        raw_ene_ids: list[list[int]] = self.get_ene_ids(document.text)
        ene_ids: list[list[int]] = []
        for raw_ene_id in raw_ene_ids:
            ene_ids.append(
                [ENE_TYPE_BIES.index("PAD")]
                + raw_ene_id[: self.max_seq_length - 2]
                + [ENE_TYPE_BIES.index("PAD")]
            )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "ene_ids": torch.tensor(ene_ids, dtype=torch.long),
            "seg_labels": torch.tensor(seg_labels, dtype=torch.long),
        }
