from dataclasses import dataclass

from kyoto_reader import BasePhrase, Document, Sentence


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
        exophors: list[str],
        kc: bool = False,
    ) -> None:
        self.kc = kc
        self.relax_exophors = {}
        for exophor in exophors:
            self.relax_exophors[exophor] = exophor
            if exophor in ("不特定:人", "不特定:物", "不特定:状況"):
                for n in "１２３４５６７８９":
                    self.relax_exophors[exophor + n] = exophor

    def _kc_skip_sentence(self, sentence: Sentence, document: Document) -> bool:
        # do not skip sentences not from Kyoto Corpus
        if self.kc is False:
            return False
        # do not skip the first sentence
        if document.doc_id.split("-")[-1] == "00":
            return False
        last_sent = document.sentences[-1] if len(document) > 0 else None
        return sentence is not last_sent

    def extract(self, document: Document, phrases: list[Phrase]):
        raise NotImplementedError

    def is_target(self, bp: BasePhrase) -> bool:
        raise NotImplementedError

    @staticmethod
    def is_candidate(bp: BasePhrase, anaphor: BasePhrase) -> bool:
        return bp.dtid < anaphor.dtid or (
            bp.dtid > anaphor.dtid and bp.sid == anaphor.sid
        )
