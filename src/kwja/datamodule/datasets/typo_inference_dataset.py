from collections import defaultdict
from typing import Dict, List, Optional

import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from kwja.utils.constants import DUMMY_TOKEN
from kwja.utils.progress_bar import track


class TypoInferenceDataset(Dataset):
    def __init__(
        self,
        texts: ListConfig,
        model_name_or_path: str = "ku-nlp/roberta-base-japanese-char-wwm",
        tokenizer_kwargs: Optional[dict] = None,
        max_seq_length: int = 512,
        **_,  # accept `extended_vocab_path` as a keyword argument
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **(tokenizer_kwargs or {}),
        )
        self.max_seq_length = max_seq_length

        self.examples: List[Dict[str, str]] = []
        self.stash: Dict[int, List[str]] = defaultdict(list)
        for text in track(texts, description="Loading documents"):
            text = text.strip()
            if len(self.tokenizer.tokenize(text)) == len(text) <= max_seq_length - 3:
                self.examples.append({"pre_text": text})
            else:
                self.stash[len(self.examples)].append(text)
        if len(self.examples) == 0:
            # len(self.examples) == 0だとwriterが呼ばれないのでダミーテキストを追加
            self.examples = [{"pre_text": ""}]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.encode(index)

    def encode(self, example_id: int) -> Dict[str, torch.Tensor]:
        example = self.examples[example_id]
        encoding: BatchEncoding = self.tokenizer(
            example["pre_text"] + DUMMY_TOKEN,
            truncation=True,
            padding=PaddingStrategy.MAX_LENGTH,
            max_length=self.max_seq_length,
        )
        return {
            "example_ids": torch.tensor(example_id, dtype=torch.long),
            "input_ids": torch.tensor(encoding.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(encoding.attention_mask, dtype=torch.long),
        }
