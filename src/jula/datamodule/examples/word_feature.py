from rhoknp import Document


class WordFeatureExample:
    """A single training/test example for word feature prediction."""

    def __init__(self) -> None:
        self.doc_id: str = ""
        self.types: list[tuple[str, str, str, str]] = []  # 形態素タイプ (品詞, 品詞細分類, 活用型, 活用形)
        self.features: list[set[str]] = []  # 形態素素性

    def load(self, document: Document) -> None:
        self.doc_id = document.doc_id
        for morpheme in document.morphemes:
            self.types.append((morpheme.pos, morpheme.subpos, morpheme.conjtype, morpheme.conjform))
        self.features = [set() for _ in document.morphemes]
        for base_phrase in document.base_phrases:
            self.features[base_phrase.head.global_index].add("基本句-主辞")
            self.features[base_phrase.morphemes[-1].global_index].add("基本句-区切")
        for phrase in document.phrases:
            self.features[phrase.morphemes[-1].global_index].add("文節-区切")
