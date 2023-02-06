from typing import List

from rhoknp import Document

from kwja.utils.constants import DISCOURSE_RELATION_MAP


class DiscourseExample:
    """A single training/test example for discourse parsing."""

    def __init__(self) -> None:
        self.doc_id: str = ""
        self.discourse_relations: List[List[str]] = []

    def load(self, document: Document, has_annotation: bool = True) -> None:
        self.doc_id = document.doc_id
        self.discourse_relations = [[""] * len(document.morphemes) for _ in range(len(document.morphemes))]
        if not has_annotation:
            return

        for modifier in document.clauses:
            modifier_morpheme_global_index = modifier.end.morphemes[0].global_index
            for head in document.clauses:
                head_morpheme_global_index = head.end.morphemes[0].global_index
                for discourse_relation in modifier.discourse_relations:
                    if discourse_relation.head == head:
                        label = DISCOURSE_RELATION_MAP[discourse_relation.label.value]
                        self.discourse_relations[modifier_morpheme_global_index][head_morpheme_global_index] = label
                if not self.discourse_relations[modifier_morpheme_global_index][head_morpheme_global_index]:
                    self.discourse_relations[modifier_morpheme_global_index][head_morpheme_global_index] = "談話関係なし"
