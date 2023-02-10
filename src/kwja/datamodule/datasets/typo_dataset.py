import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from kwja.utils.constants import DUMMY_TOKEN, IGNORE_INDEX, TYPO_CORR_OP_TAG2TOKEN
from kwja.utils.progress_bar import track
from kwja.utils.typo_module_writer import get_maps


class TypoDataset(Dataset):
    def __init__(
        self,
        path: str,
        extended_vocab_path: str,
        model_name_or_path: str = "ku-nlp/roberta-base-japanese-char-wwm",
        tokenizer_kwargs: Optional[dict] = None,
        max_seq_length: int = 512,
    ) -> None:
        self.path = Path(path)
        assert self.path.is_dir()

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **(tokenizer_kwargs or {}),
        )
        assert self.tokenizer.unk_token_id is not None
        self.max_seq_length: int = max_seq_length

        self.token2token_id, self.token_id2token = get_maps(self.tokenizer, extended_vocab_path)

        self.examples = self.load_examples(self.path)
        self.stash: Dict[int, List[str]] = defaultdict(list)
        assert len(self) > 0

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.encode(index)

    @staticmethod
    def load_examples(example_dir: Path) -> List[Dict[str, Union[str, List[str]]]]:
        examples: List[Dict[str, Union[str, List[str]]]] = []
        for path in track(sorted(example_dir.glob("**/*.jsonl")), description="Loading documents"):
            for line in path.read_text().strip().split("\n"):
                examples.append(json.loads(line))
        return examples

    def encode(self, example_id: int) -> Dict[str, torch.Tensor]:
        example: Dict[str, Union[str, List[str]]] = self.examples[example_id]

        assert type(example["pre_text"]) == str, "type of pre_text is invalid"
        encoding: BatchEncoding = self.tokenizer(
            example["pre_text"] + DUMMY_TOKEN,
            truncation=True,
            padding=PaddingStrategy.MAX_LENGTH,
            max_length=self.max_seq_length,
        )

        kdr_labels: List[int] = []
        for kdr_tag in example["kdrs"][:-1]:
            if kdr_tag in TYPO_CORR_OP_TAG2TOKEN:
                kdr_label = self.token2token_id[TYPO_CORR_OP_TAG2TOKEN[kdr_tag]]
            else:
                # remove prefix "R:" from kdr tag
                kdr_label = self.token2token_id.get(kdr_tag[2:], self.tokenizer.unk_token_id)
            kdr_labels.append(kdr_label)
        kdr_labels.append(IGNORE_INDEX)
        # padding
        kdr_labels = [IGNORE_INDEX] + kdr_labels[: self.max_seq_length - 2] + [IGNORE_INDEX]
        kdr_labels += [IGNORE_INDEX] * (self.max_seq_length - len(kdr_labels))

        ins_labels: List[int] = []
        for ins_tag in example["inss"]:
            if ins_tag in TYPO_CORR_OP_TAG2TOKEN:
                ins_label = self.token2token_id[TYPO_CORR_OP_TAG2TOKEN[ins_tag]]
            else:
                # remove prefix "I:" from ins tag
                ins_label = self.token2token_id.get(ins_tag[2:], self.tokenizer.unk_token_id)
            ins_labels.append(ins_label)
        # padding
        ins_labels = [IGNORE_INDEX] + ins_labels[: self.max_seq_length - 2] + [IGNORE_INDEX]
        ins_labels += [IGNORE_INDEX] * (self.max_seq_length - len(ins_labels))

        return {
            "example_ids": torch.tensor(example_id, dtype=torch.long),
            "input_ids": torch.tensor(encoding.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(encoding.attention_mask, dtype=torch.long),
            "kdr_labels": torch.tensor(kdr_labels, dtype=torch.long),
            "ins_labels": torch.tensor(ins_labels, dtype=torch.long),
        }
