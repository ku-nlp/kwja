from dataclasses import dataclass
from typing import List

from transformers import BatchEncoding


@dataclass(frozen=True)
class Seq2SeqExample:
    example_id: int
    src_text: str
    src_encoding: BatchEncoding
    tgt_input_ids: List[int]
    tgt_attention_mask: List[int]
    sid: str


@dataclass(frozen=True)
class Seq2SeqInferenceExample:
    example_id: int
    src_text: str
    src_encoding: BatchEncoding
    sid: str
