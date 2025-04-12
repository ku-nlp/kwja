from dataclasses import dataclass


@dataclass(frozen=True)
class TypoExample:
    example_id: int
    doc_id: str
    pre_text: str
    post_text: str
    kdr_tags: list[str]
    ins_tags: list[str]


@dataclass(frozen=True)
class TypoInferenceExample:
    example_id: int
    doc_id: str
    pre_text: str
