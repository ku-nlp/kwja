from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TypoExample:
    example_id: int
    pre_text: str
    post_text: str
    kdrs: List[str]
    inss: List[str]


@dataclass(frozen=True)
class TypoInferenceExample:
    example_id: int
    pre_text: str
