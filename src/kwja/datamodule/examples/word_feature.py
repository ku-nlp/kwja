from rhoknp import Document
from rhoknp.props import NamedEntity

from kwja.utils.constants import SUB_WORD_FEATURES
from kwja.utils.sub_document import extract_target_sentences


class WordFeatureExample:
    """A single training/test example for word feature prediction."""

    def __init__(self) -> None:
        self.doc_id: str = ""
        self.types: dict[int, tuple[str, str, str, str]] = {}  # 形態素 index -> 形態素タイプ (品詞, 品詞細分類, 活用型, 活用形)
        self.named_entities: list[NamedEntity] = []  # 固有表現
        self.features: dict[int, set[str]] = {}  # 形態素 index -> 形態素素性集合

    def load(self, document: Document) -> None:
        self.doc_id = document.doc_id
        for sentence in extract_target_sentences(document):
            for morpheme in sentence.morphemes:
                self.types[morpheme.global_index] = (
                    morpheme.pos,
                    morpheme.subpos,
                    morpheme.conjtype,
                    morpheme.conjform,
                )

            self.named_entities += sentence.named_entities

            for morpheme in sentence.morphemes:
                self.features[morpheme.global_index] = set()
                for sub_word_feature in SUB_WORD_FEATURES:
                    if sub_word_feature in morpheme.features:
                        self.features[morpheme.global_index].add(sub_word_feature)
            for base_phrase in sentence.base_phrases:
                self.features[base_phrase.head.global_index].add("基本句-主辞")
                self.features[base_phrase.morphemes[-1].global_index].add("基本句-区切")
            for phrase in sentence.phrases:
                self.features[phrase.morphemes[-1].global_index].add("文節-区切")
