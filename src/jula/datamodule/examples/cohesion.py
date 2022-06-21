import logging
from enum import Enum
from typing import Optional

from rhoknp import Document
from tokenizers import Encoding

from jula.datamodule.extractors import (
    Annotation,
    BridgingExtractor,
    CoreferenceExtractor,
    PasExtractor,
)
from jula.datamodule.extractors.base import Extractor, Mrph, Phrase

logger = logging.getLogger(__file__)


class Task(Enum):
    PAS_ANALYSIS = "pas_analysis"
    BRIDGING = "bridging"
    COREFERENCE = "coreference"


TASK2EXTRACTOR = {
    Task.PAS_ANALYSIS: PasExtractor,
    Task.BRIDGING: BridgingExtractor,
    Task.COREFERENCE: CoreferenceExtractor,
}


class CohesionExample:
    """A single training/test example for bridging anaphora resolution."""

    def __init__(self) -> None:
        self.example_id: int = -1
        self.doc_id: str = ""
        self.annotations: dict[Task, Annotation] = {}
        self.phrases: dict[Task, list[Phrase]] = {}
        self.encoding: Optional[Encoding] = None

    @property
    def mrphs(self) -> dict[Task, list[Mrph]]:
        return {
            task: sum((p.children for p in phrases), [])
            for task, phrases in self.phrases.items()
        }

    def load(
        self,
        document: Document,
        tasks: list[Task],
        extractors: dict[Task, Extractor],
    ):
        self.doc_id = document.doc_id
        for task in tasks:
            phrases = self._construct_phrases(document)
            extractor: Extractor = extractors[task]
            self.phrases[task] = phrases
            self.annotations[task] = extractor.extract(
                document, phrases
            )  # extract gold

    @staticmethod
    def _construct_phrases(document: Document) -> list[Phrase]:
        # construct phrases and mrphs
        phrases = []
        for sentence in document.sentences:
            for anaphor in sentence.base_phrases:
                phrase = Phrase(
                    dtid=anaphor.global_index,
                    surf=anaphor.text,
                    dmids=[m.global_index for m in anaphor.morphemes],
                    dmid=anaphor.head.global_index,
                    children=[],
                    is_target=False,
                    candidates=[],
                )
                for morpheme in anaphor.morphemes:
                    mrph = Mrph(
                        dmid=morpheme.global_index,
                        surf=morpheme.text,
                        parent=phrase,
                    )
                    phrase.children.append(mrph)
                phrases.append(phrase)
        return phrases

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        string = ""
        for i, phrase in enumerate(next(iter(self.phrases.values()))):
            string += f"{i:02} {phrase.surf}"
            string += " " * (6 - len(phrase.surf)) * 2
            if ann := self.annotations.get(Task.PAS_ANALYSIS):
                for case, args in ann.arguments_set[i].items():
                    args_str = ",".join(str(arg) for arg in args)
                    string += f"{case}:{args_str:6}  "
            if ann := self.annotations.get(Task.BRIDGING):
                args_str = ",".join(str(arg) for arg in ann.arguments_set[i])
                string += f"ノ:{args_str:6}  "
            if ann := self.annotations.get(Task.COREFERENCE):
                args_str = ",".join(str(arg) for arg in ann.arguments_set[i])
                string += f"＝:{args_str:6}"
            string += "\n"
        return string
