import hydra
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizer
from transformers.utils import PaddingStrategy


class SegmentedTextDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        model_name_or_path: str = "nlp-waseda/roberta-base-japanese",
        max_seq_length: int = 512,
        tokenizer_kwargs: dict = None,
    ) -> None:
        self.texts = [text.strip() for text in texts]
        if tokenizer_kwargs:
            tokenizer_kwargs = hydra.utils.instantiate(tokenizer_kwargs, _convert_="partial")
        else:
            tokenizer_kwargs = {}
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **tokenizer_kwargs,
        )
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "example_ids": torch.tensor(index, dtype=torch.long),
            **self.encode(self.texts[index]),
        }

    def encode(self, text: str) -> dict[str, torch.Tensor]:
        encoding: BatchEncoding = self.tokenizer(
            text,
            truncation=True,
            padding=PaddingStrategy.MAX_LENGTH,
            max_length=self.max_seq_length - 1,
        )
        input_ids = encoding["input_ids"] + [self.tokenizer.vocab["[ROOT]"]]
        attention_mask = encoding["attention_mask"] + [1]
        subword_map = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        for token_id, word_id in enumerate(encoding.word_ids()):
            if word_id is not None:
                subword_map[word_id][token_id] = True
        intra_mask = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        num_morphemes = len(text.split(" "))
        for i in range(0, num_morphemes):
            for j in range(0, num_morphemes):
                if i != j:
                    intra_mask[i][j] = True
            intra_mask[i][-1] = True
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "subword_map": torch.tensor(subword_map, dtype=torch.bool),
            "intra_mask": torch.tensor(intra_mask, dtype=torch.bool),
            "texts": text,
        }
