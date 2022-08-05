import hydra
import torch
from rhoknp.cohesion import ExophoraReferent
from tokenizers import Encoding
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from jula.datamodule.examples import Task


class WordInferenceDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        model_name_or_path: str = "nlp-waseda/roberta-base-japanese",
        max_seq_length: int = 512,
        tokenizer_kwargs: dict = None,
    ) -> None:
        self.texts = [text.strip() for text in texts]
        self.exophora_referents: list[ExophoraReferent] = [ExophoraReferent(s) for s in ["著者", "読者", "不特定:人", "不特定:物"]]
        self.special_tokens: list[str] = [str(e) for e in self.exophora_referents] + [
            "[NULL]",
            "[NA]",
            "[ROOT]",  # TODO: mask in cohesion analysis
        ]
        if tokenizer_kwargs:
            tokenizer_kwargs = hydra.utils.instantiate(tokenizer_kwargs, _convert_="partial")
        else:
            tokenizer_kwargs = {}
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **tokenizer_kwargs,
        )
        self.max_seq_length = max_seq_length
        self.special_to_index: dict[str, int] = {
            token: self.max_seq_length - len(self.special_tokens) + i for i, token in enumerate(self.special_tokens)
        }
        self.cohesion_tasks: list[Task] = [Task(t) for t in ["pas_analysis", "bridging", "coreference"]]
        self.rels = ["ガ", "ヲ", "ニ", "ガ２", "ノ", "="]

    @property
    def special_indices(self) -> list[int]:
        return list(self.special_to_index.values())

    @property
    def num_special_tokens(self) -> int:
        return len(self.special_tokens)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.encode(self.texts[index], index)

    def encode(self, text: str, example_id: int) -> dict[str, torch.Tensor]:
        encoding: Encoding = self.tokenizer(
            text,
            truncation=True,
            padding=PaddingStrategy.MAX_LENGTH,
            max_length=self.max_seq_length - self.num_special_tokens,
        ).encodings[0]

        intra_mask = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        num_morphemes = len(text.split(" "))
        for i in range(0, num_morphemes):
            for j in range(0, num_morphemes):
                if i != j:
                    intra_mask[i][j] = True
            intra_mask[i][-1] = True
        cohesion_mask = [True] * num_morphemes
        cohesion_mask += [False] * (self.max_seq_length - num_morphemes - self.num_special_tokens)
        cohesion_mask += [True] * self.num_special_tokens

        special_encoding: Encoding = self.tokenizer(
            self.special_tokens,
            is_split_into_words=True,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=False,
            add_special_tokens=False,
        ).encodings[0]
        merged_encoding: Encoding = Encoding.merge([encoding, special_encoding])

        return {
            "example_ids": torch.tensor(example_id, dtype=torch.long),
            "input_ids": torch.tensor(merged_encoding.ids, dtype=torch.long),
            "attention_mask": torch.tensor(merged_encoding.attention_mask, dtype=torch.long),
            "subword_map": torch.tensor(self._gen_subword_map(merged_encoding), dtype=torch.bool),
            "intra_mask": torch.tensor(intra_mask, dtype=torch.bool),
            "cohesion_mask": torch.tensor(cohesion_mask, dtype=torch.bool)
            .view(1, 1, -1)
            .expand(len(self.rels), self.max_seq_length, self.max_seq_length),
            "texts": text,
        }

    def _gen_subword_map(self, encoding: Encoding) -> list[list[bool]]:
        subword_map = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        for token_id, word_id in enumerate(encoding.word_ids):
            if word_id is not None:
                subword_map[word_id][token_id] = True
        for special_index in self.special_indices:
            subword_map[special_index][special_index] = True
        return subword_map
