from dataclasses import dataclass
from typing import List


@dataclass
class Seq2SeqExample:
    example_id: int
    surfs: List[str]
    src_input_ids: List[int]
    src_attention_mask: List[int]
    tgt_input_ids: List[int]
    sid: str


@dataclass(frozen=True)
class Seq2SeqInferenceExample:
    example_id: int
    surfs: List[str]
    src_input_ids: List[int]
    src_attention_mask: List[int]
    sid: str
