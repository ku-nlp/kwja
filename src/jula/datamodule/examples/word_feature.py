from collections import defaultdict

from rhoknp import Document
from rhoknp.props import NamedEntity

from jula.utils.constants import SUB_DOC_PAT, SUB_WORD_FEATURES


class WordFeatureExample:
    """A single training/test example for word feature prediction."""

    def __init__(self) -> None:
        self.doc_id: str = ""
        self.types: list[tuple[int, tuple[str, str, str, str]]] = []  # 形態素タイプ (品詞, 品詞細分類, 活用型, 活用形)
        self.named_entities: list[NamedEntity] = []  # 固有表現
        self.features: dict[int, set[str]] = defaultdict(set)  # 形態素素性

    def load(self, document: Document) -> None:
        self.doc_id = document.doc_id
        if SUB_DOC_PAT.search(self.doc_id) is not None:
            target = document.sentences[-1]
        else:
            target = document

        for morpheme in target.morphemes:
            types = (morpheme.pos, morpheme.subpos, morpheme.conjtype, morpheme.conjform)
            self.types.append((morpheme.global_index, types))

        self.named_entities = target.named_entities

        self.features = defaultdict(set)
        for morpheme in target.morphemes:
            for sub_word_feature in SUB_WORD_FEATURES:
                if sub_word_feature in morpheme.features:
                    self.features[morpheme.global_index].add(sub_word_feature)
        for base_phrase in target.base_phrases:
            self.features[base_phrase.head.global_index].add("基本句-主辞")
            self.features[base_phrase.morphemes[-1].global_index].add("基本句-区切")
        for phrase in target.phrases:
            self.features[phrase.morphemes[-1].global_index].add("文節-区切")
