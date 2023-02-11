import logging
from typing import Dict, List

from rhoknp import Document

from kwja.utils.cohesion_analysis import CohesionBasePhrase, CohesionUtils
from kwja.utils.constants import CohesionTask

logger = logging.getLogger(__name__)


class CohesionExample:
    """A single training/test example for cohesion analysis."""

    def __init__(self) -> None:
        self.doc_id: str = ""
        self.task2base_phrases: Dict[CohesionTask, List[CohesionBasePhrase]] = {}

    def load(
        self,
        document: Document,
        cohesion_task2utils: Dict[CohesionTask, CohesionUtils],
    ):
        self.doc_id = document.doc_id
        base_phrases = document.base_phrases
        for cohesion_task, cohesion_utils in cohesion_task2utils.items():
            self.task2base_phrases[cohesion_task] = cohesion_utils.wrap(base_phrases)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        string = ""
        for i, cohesion_base_phrase in enumerate(list(self.task2base_phrases.values())[0]):
            string += f"{i:02} {cohesion_base_phrase.text}"
            string += " " * (6 - len(cohesion_base_phrase.text)) * 2
            if cohesion_base_phrases := self.task2base_phrases.get(CohesionTask.PAS_ANALYSIS):
                for case, argument_tags in cohesion_base_phrases[i].relation2tags.items():
                    string += f"{case}:{','.join(argument_tags):6}  "
            if cohesion_base_phrases := self.task2base_phrases.get(CohesionTask.BRIDGING_REFERENCE_RESOLUTION):
                string += f"ノ:{','.join(cohesion_base_phrases[i].relation2tags['ノ']):6}  "
            if cohesion_base_phrases := self.task2base_phrases.get(CohesionTask.COREFERENCE_RESOLUTION):
                string += f"=:{','.join(cohesion_base_phrases[i].relation2tags['=']):6}"
            string += "\n"
        return string
