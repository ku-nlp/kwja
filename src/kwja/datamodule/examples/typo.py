from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TypoExample:
    example_id: int
    doc_id: str
    pre_text: str
    post_text: str
    kdr_tags: List[str]
    ins_tags: List[str]


@dataclass(frozen=True)
class TypoInferenceExample:
    example_id: int
    doc_id: str
    pre_text: str
