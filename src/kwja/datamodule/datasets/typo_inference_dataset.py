from collections import defaultdict
from typing import Dict, List, Union

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from kwja.utils.constants import TYPO_DUMMY_TOKEN
from kwja.utils.progress_bar import track


class TypoInferenceDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        model_name_or_path: str = "ku-nlp/roberta-base-japanese-char-wwm",
        max_seq_length: int = 512,
        tokenizer_kwargs: dict = None,
        **_,  # accept `extended_vocab_path` as a keyword argument
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **(tokenizer_kwargs or {}),
        )
        self.max_seq_length = max_seq_length
        self.texts = []
        self.stash: Dict[int, List[str]] = defaultdict(list)
        for text in track(texts, description="Loading documents"):
            text = text.strip()
            if len(self.tokenizer.tokenize(text)) == len(text) <= max_seq_length - 3:
                self.texts.append(text)
            else:
                self.stash[len(self.texts)].append(text)
        if len(self.texts) == 0:
            self.texts = [""]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        return self.encode(index)

    def encode(self, example_id: int) -> Dict[str, Union[torch.Tensor, str]]:
        text = self.texts[example_id]
        encoding: BatchEncoding = self.tokenizer(
            text + TYPO_DUMMY_TOKEN,
            truncation=True,
            padding=PaddingStrategy.MAX_LENGTH,
            max_length=self.max_seq_length,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        return {
            "example_ids": torch.tensor(example_id, dtype=torch.long),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "texts": text,
        }
