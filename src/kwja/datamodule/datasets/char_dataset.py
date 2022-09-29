import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
from rhoknp import Document, Sentence
from tqdm import tqdm
from transformers import BatchEncoding
from transformers.utils import PaddingStrategy

from kwja.datamodule.datasets.base_dataset import BaseDataset
from kwja.datamodule.examples.char_feature import CharFeatureExample
from kwja.utils.constants import IGNORE_INDEX
from kwja.utils.word_normalize import SentenceDenormalizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CharExampleSet:
    example_id: int
    doc_id: str
    text: str  # space-delimited word sequence
    encoding: BatchEncoding
    char_feature_example: CharFeatureExample


class CharDataset(BaseDataset):
    def __init__(
        self,
        path: str,
        document_split_stride: int,
        model_name_or_path: str = "ku-nlp/roberta-base-japanese-char-wwm",
        max_seq_length: int = 512,
        tokenizer_kwargs: dict = None,
        denormalize_prob: float = 0.0,
    ) -> None:
        self.path = Path(path)
        super().__init__(
            self.path,
            document_split_stride,
            model_name_or_path,
            max_seq_length,
            tokenizer_kwargs or {},
        )
        self.denormalizer: SentenceDenormalizer = SentenceDenormalizer()
        self.denormalize_prob: float = denormalize_prob
        self.examples: list[CharExampleSet] = self._load_examples(self.documents)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.encode(self.examples[index])

    def _load_examples(self, documents: list[Document]) -> list[CharExampleSet]:
        examples = []
        idx = 0
        for document in tqdm(documents, dynamic_ncols=True):
            for sentence in document.sentences:
                self.denormalizer.denormalize(sentence, self.denormalize_prob)
            encoding: BatchEncoding = self.tokenizer(
                document.text,
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=False,
                max_length=self.max_seq_length,
            )
            if len(encoding.input_ids) > self.max_seq_length:
                continue

            char_feature_example = CharFeatureExample()
            char_feature_example.load(document)

            examples.append(
                CharExampleSet(
                    example_id=idx,
                    doc_id=document.doc_id,
                    text=document.text,
                    encoding=encoding,
                    char_feature_example=char_feature_example,
                )
            )
            idx += 1

        if len(examples) == 0:
            logger.error(
                "No examples to process. "
                f"Make sure there exist any documents in {self.path} and they are not too long."
            )
        return examples

    def encode(self, example: CharExampleSet) -> dict[str, torch.Tensor]:
        char_feature_example = example.char_feature_example
        seg_types: list[int] = [IGNORE_INDEX for _ in range(self.max_seq_length)]
        for i, seg_label in char_feature_example.seg_types.items():
            # 先頭のCLSトークンをIGNORE_INDEXにするため+1
            seg_types[i + 1] = seg_label
        norm_types: list[int] = [IGNORE_INDEX for _ in range(self.max_seq_length)]
        for i, norm_label in char_feature_example.norm_types.items():
            # 先頭のCLSトークンをIGNORE_INDEXにするため+1
            norm_types[i + 1] = norm_label

        return {
            "example_ids": torch.tensor(example.example_id, dtype=torch.long),
            "input_ids": torch.tensor(example.encoding.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(example.encoding.attention_mask, dtype=torch.long),
            "seg_types": torch.tensor(seg_types, dtype=torch.long),
            "norm_types": torch.tensor(norm_types, dtype=torch.long),
        }

    def _get_tokenized_len(self, source: Union[Document, Sentence]) -> int:
        return len(self.tokenizer(source.text, add_special_tokens=False)["input_ids"])
