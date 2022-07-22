from rhoknp import Document


class WordFeatureExample:
    """A single training/test example for word feature prediction."""

    def __init__(self) -> None:
        self.example_id: int = -1
        self.doc_id: str = ""
        self.features: list[set[str]] = []  # 形態素単位素性リスト

    def load(self, document: Document) -> None:
        self.doc_id = document.doc_id
        self.features = [set() for _ in document.morphemes]
        for base_phrase in document.base_phrases:
            self.features[base_phrase.head.global_index].add("基本句-主辞")
            self.features[base_phrase.morphemes[-1].global_index].add("基本句-区切")
        for phrase in document.phrases:
            self.features[phrase.morphemes[-1].global_index].add("文節-区切")
