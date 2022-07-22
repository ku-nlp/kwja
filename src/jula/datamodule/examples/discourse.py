from rhoknp import Document


class DiscourseExample:
    """A single training/test example for discourse parsing."""

    def __init__(self) -> None:
        self.example_id: int = -1
        self.doc_id: str = ""
        self.discourse_relations: list[list[str]] = []

    def load(self, document: Document) -> None:
        self.doc_id = document.doc_id
        self.discourse_relations = [[""] * len(document.morphemes) for _ in range(len(document.morphemes))]
        for modifier in document.clauses:
            modifier_morpheme_id = modifier.end.morphemes[0].global_index
            for head in document.clauses:
                head_morpheme_id = head.end.morphemes[0].global_index
                for discourse_relation in modifier.discourse_relations:
                    if discourse_relation.head == head:
                        label = discourse_relation.label
                        self.discourse_relations[modifier_morpheme_id][head_morpheme_id] = label
                else:
                    if not self.discourse_relations[modifier_morpheme_id][head_morpheme_id]:
                        self.discourse_relations[modifier_morpheme_id][head_morpheme_id] = "談話関係なし"
