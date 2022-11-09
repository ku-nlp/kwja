from typing import Dict, List, Union

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from kwja.utils.constants import TYPO_DUMMY_TOKEN


class TypoInferenceDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        model_name_or_path: str = "nlp-waseda/roberta-base-japanese",
        max_seq_length: int = 512,
        tokenizer_kwargs: dict = None,
        **_,  # accept `extended_vocab_path` as a keyword argument
    ) -> None:
        self.texts = [text.strip() for text in texts]
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **(tokenizer_kwargs or {}),
        )
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        return self.encode(self.texts[index])

    def encode(self, text: str) -> Dict[str, Union[torch.Tensor, str]]:
        encoding: BatchEncoding = self.tokenizer(
            text + TYPO_DUMMY_TOKEN,
            truncation=True,
            padding=PaddingStrategy.MAX_LENGTH,
            max_length=self.max_seq_length,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "texts": text,
        }
