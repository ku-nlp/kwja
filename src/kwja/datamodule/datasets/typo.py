import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from torch.utils.data import Dataset
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from kwja.callbacks.utils import get_maps
from kwja.datamodule.examples import TypoExample
from kwja.utils.constants import DUMMY_TOKEN, IGNORE_INDEX, RESOURCE_PATH, TYPO_CORR_OP_TAG2TOKEN
from kwja.utils.progress_bar import track


@dataclass(frozen=True)
class TypoModuleFeatures:
    example_ids: int
    input_ids: List[int]
    attention_mask: List[int]
    kdr_labels: List[int]
    ins_labels: List[int]


class TypoDataset(Dataset[TypoModuleFeatures]):
    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
    ) -> None:
        self.path = Path(path)
        assert self.path.is_dir()

        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        assert self.tokenizer.unk_token_id is not None
        self.max_seq_length: int = max_seq_length

        self.token2token_id, self.token_id2token = get_maps(
            self.tokenizer,
            RESOURCE_PATH / "typo_correction" / "multi_char_vocab.txt",
        )

        self.examples: List[TypoExample] = self._load_examples(self.path)
        self.stash: Dict[int, List[str]] = defaultdict(list)
        assert len(self) > 0

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TypoModuleFeatures:
        return self.encode(self.examples[index])

    @staticmethod
    def _load_examples(example_dir: Path) -> List[TypoExample]:
        examples: List[TypoExample] = []
        example_id = 0
        for path in track(sorted(example_dir.glob("**/*.jsonl")), description="Loading documents"):
            for line in path.read_text().strip().split("\n"):
                examples.append(TypoExample(**json.loads(line), example_id=example_id))
                example_id += 1
        return examples

    def encode(self, example: TypoExample) -> TypoModuleFeatures:
        encoding: BatchEncoding = self.tokenizer(
            example.pre_text + DUMMY_TOKEN,
            truncation=True,
            padding=PaddingStrategy.MAX_LENGTH,
            max_length=self.max_seq_length,
        )

        kdr_labels: List[int] = []
        for kdr_tag in example.kdrs[:-1]:
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
        for ins_tag in example.inss:
            if ins_tag in TYPO_CORR_OP_TAG2TOKEN:
                ins_label = self.token2token_id[TYPO_CORR_OP_TAG2TOKEN[ins_tag]]
            else:
                # remove prefix "I:" from ins tag
                ins_label = self.token2token_id.get(ins_tag[2:], self.tokenizer.unk_token_id)
            ins_labels.append(ins_label)
        # padding
        ins_labels = [IGNORE_INDEX] + ins_labels[: self.max_seq_length - 2] + [IGNORE_INDEX]
        ins_labels += [IGNORE_INDEX] * (self.max_seq_length - len(ins_labels))

        return TypoModuleFeatures(
            example_ids=example.example_id,
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            kdr_labels=kdr_labels,
            ins_labels=ins_labels,
        )