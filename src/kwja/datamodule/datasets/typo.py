import json
from collections import defaultdict
from dataclasses import dataclass
from importlib.resources import as_file
from pathlib import Path

from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from kwja.datamodule.datasets.base import BaseDataset
from kwja.datamodule.examples import TypoExample
from kwja.utils.constants import DUMMY_TOKEN, IGNORE_INDEX, RESOURCE_TRAVERSABLE, TYPO_CORR_OP_TAG2TOKEN
from kwja.utils.logging_util import track

MULTI_CHAR_VOCAB_TRAVERSABLE = RESOURCE_TRAVERSABLE / "typo_correction" / "multi_char_vocab.txt"


def get_maps(tokenizer: PreTrainedTokenizerBase) -> tuple[dict[str, int], dict[int, str]]:
    token2token_id = tokenizer.get_vocab()
    with as_file(MULTI_CHAR_VOCAB_TRAVERSABLE) as path:
        with open(path) as f:
            for line in f:
                if line := line.strip():
                    token2token_id[line] = len(token2token_id.keys())
    token_id2token = {v: k for k, v in token2token_id.items()}
    return token2token_id, token_id2token


@dataclass(frozen=True)
class TypoModuleFeatures:
    example_ids: int
    input_ids: list[int]
    attention_mask: list[int]
    kdr_labels: list[int]
    ins_labels: list[int]


class TypoDataset(BaseDataset[TypoExample, TypoModuleFeatures]):
    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
    ) -> None:
        super().__init__(tokenizer, max_seq_length)

        self.path = Path(path)
        assert self.path.is_dir()

        assert self.tokenizer.unk_token_id is not None

        self.token2token_id, self.token_id2token = get_maps(self.tokenizer)

        self.examples: list[TypoExample] = self._load_examples(self.path)
        self.stash: dict[int, list[tuple[str, str]]] = defaultdict(list)
        assert len(self) > 0

    @staticmethod
    def _load_examples(example_dir: Path) -> list[TypoExample]:
        examples: list[TypoExample] = []
        example_id = 0
        for path in track(sorted(example_dir.glob("**/*.jsonl")), description="Loading documents"):
            for line in path.read_text().strip().split("\n"):
                examples.append(TypoExample(**json.loads(line), example_id=example_id, doc_id=""))
                example_id += 1
        return examples

    def encode(self, example: TypoExample) -> TypoModuleFeatures:
        encoding: BatchEncoding = self.tokenizer(
            example.pre_text + DUMMY_TOKEN,
            truncation=True,
            padding=PaddingStrategy.MAX_LENGTH,
            max_length=self.max_seq_length,
        )

        kdr_labels: list[int] = []
        for kdr_tag in example.kdr_tags[:-1]:
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

        ins_labels: list[int] = []
        for ins_tag in example.ins_tags:
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
