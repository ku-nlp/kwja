from rhoknp import Document

from kwja.utils.constants import DISCOURSE_RELATION_MAP


class DiscourseExample:
    """A single training/test example for discourse parsing."""

    def __init__(self) -> None:
        self.doc_id: str = ""
        self.discourse_relations: list[list[str]] = []

    def load(self, document: Document, has_annotation: bool = True) -> None:
        self.doc_id = document.doc_id
        self.discourse_relations = [[""] * len(document.morphemes) for _ in range(len(document.morphemes))]
        if not has_annotation:
            return
        for modifier in document.clauses:
            modifier_morpheme_id = modifier.end.morphemes[0].global_index
            for head in document.clauses:
                head_morpheme_id = head.end.morphemes[0].global_index
                for discourse_relation in modifier.discourse_relations:
                    if discourse_relation.head == head:
                        label = DISCOURSE_RELATION_MAP[discourse_relation.label.value]
                        self.discourse_relations[modifier_morpheme_id][head_morpheme_id] = label
                if not self.discourse_relations[modifier_morpheme_id][head_morpheme_id]:
                    self.discourse_relations[modifier_morpheme_id][head_morpheme_id] = "談話関係なし"
