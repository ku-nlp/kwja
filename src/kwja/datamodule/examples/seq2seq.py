from dataclasses import dataclass


@dataclass
class Seq2SeqExample:
    example_id: int
    surfs: list[str]
    src_input_ids: list[int]
    src_attention_mask: list[int]
    tgt_input_ids: list[int]
    sid: str


@dataclass(frozen=True)
class Seq2SeqInferenceExample:
    example_id: int
    surfs: list[str]
    src_input_ids: list[int]
    src_attention_mask: list[int]
    sid: str
