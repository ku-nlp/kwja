import logging
from enum import Enum

from rhoknp import Document

from kwja.datamodule.extractors import Annotation, BridgingExtractor, CoreferenceExtractor, PasExtractor
from kwja.datamodule.extractors.base import Extractor, Mrph, Phrase

logger = logging.getLogger(__name__)


class CohesionTask(Enum):
    PAS_ANALYSIS = "pas_analysis"
    BRIDGING = "bridging"
    COREFERENCE = "coreference"


TASK2EXTRACTOR = {
    CohesionTask.PAS_ANALYSIS: PasExtractor,
    CohesionTask.BRIDGING: BridgingExtractor,
    CohesionTask.COREFERENCE: CoreferenceExtractor,
}


class CohesionExample:
    """A single training/test example for cohesion analysis."""

    def __init__(self) -> None:
        self.doc_id: str = ""
        self.annotations: dict[CohesionTask, Annotation] = {}
        self.phrases: dict[CohesionTask, list[Phrase]] = {}

    @property
    def mrphs(self) -> dict[CohesionTask, list[Mrph]]:
        return {task: sum((p.children for p in phrases), []) for task, phrases in self.phrases.items()}

    def load(
        self,
        document: Document,
        tasks: list[CohesionTask],
        extractors: dict[CohesionTask, Extractor],
    ):
        self.doc_id = document.doc_id
        for task in tasks:
            phrases = self._construct_phrases(document)
            extractor: Extractor = extractors[task]
            self.phrases[task] = phrases
            self.annotations[task] = extractor.extract(document, phrases)  # extract gold

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
            if ann := self.annotations.get(CohesionTask.PAS_ANALYSIS):
                for case, args in ann.arguments_set[i].items():
                    args_str = ",".join(str(arg) for arg in args)
                    string += f"{case}:{args_str:6}  "
            if ann := self.annotations.get(CohesionTask.BRIDGING):
                args_str = ",".join(str(arg) for arg in ann.arguments_set[i])
                string += f"ノ:{args_str:6}  "
            if ann := self.annotations.get(CohesionTask.COREFERENCE):
                args_str = ",".join(str(arg) for arg in ann.arguments_set[i])
                string += f"＝:{args_str:6}"
            string += "\n"
        return string
