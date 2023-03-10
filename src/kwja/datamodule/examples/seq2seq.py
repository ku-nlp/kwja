from dataclasses import dataclass

from transformers import BatchEncoding


@dataclass(frozen=True)
class Seq2SeqExample:
    example_id: int
    src_text: str
    src_encoding: BatchEncoding
    tgt_encoding: BatchEncoding
    sid: str


@dataclass(frozen=True)
class Seq2SeqInferenceExample:
    example_id: int
    src_text: str
    src_encoding: BatchEncoding
    sid: str
