from dataclasses import dataclass

from rhoknp import BasePhrase, Document
from rhoknp.cohesion import ExophoraReferent, ExophoraReferentType


@dataclass(frozen=True)
class Mrph:
    dmid: int
    surf: str
    parent: "Phrase"

    @property
    def is_target(self):
        return self.parent.is_target and self.parent.dmid == self.dmid


@dataclass(frozen=False)
class Phrase:
    dtid: int
    surf: str
    dmids: list[int]
    dmid: int
    children: list[Mrph]
    is_target: bool
    candidates: list[int]  # candidates list is always empty if is_target is False


class Extractor:
    def __init__(
        self,
        exophora_referents: list[ExophoraReferent],
        restrict_target: bool = False,
    ) -> None:
        self.exophora_referents = exophora_referents
        self.restrict_target = restrict_target

    @staticmethod
    def _relax_exophora_referent(
        exophora_referent: ExophoraReferent,
    ) -> ExophoraReferent:
        if exophora_referent.type in (
            ExophoraReferentType.UNSPECIFIED_PERSON,
            ExophoraReferentType.UNSPECIFIED_MATTER,
            ExophoraReferentType.UNSPECIFIED_SITUATION,
        ):
            exophora_referent.index = None
        return exophora_referent

    def extract(self, document: Document, phrases: list[Phrase]):
        raise NotImplementedError

    def is_target(self, bp: BasePhrase) -> bool:
        raise NotImplementedError

    @staticmethod
    def is_candidate(bp: BasePhrase, anaphor: BasePhrase) -> bool:
        return bp.global_index < anaphor.global_index or (
            bp.global_index > anaphor.global_index and bp.sentence.sid == anaphor.sentence.sid
        )
