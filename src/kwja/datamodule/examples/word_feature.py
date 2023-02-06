from typing import Dict, List, Set, Tuple

from rhoknp import Document
from rhoknp.props import NamedEntity

from kwja.utils.constants import SUB_WORD_FEATURES
from kwja.utils.sub_document import extract_target_sentences


class WordFeatureExample:
    """A single training/test example for word feature prediction."""

    def __init__(self) -> None:
        self.doc_id: str = ""
        # 形態素 global index -> 形態素の属性 (品詞, 品詞細分類, 活用型, 活用形)
        self.global_index2attributes: Dict[int, Tuple[str, str, str, str]] = {}
        self.global_index2feature_set: Dict[int, Set[str]] = {}  # 形態素 global index -> 素性集合
        self.named_entities: List[NamedEntity] = []  # 固有表現

    def load(self, document: Document) -> None:
        self.doc_id = document.doc_id
        for sentence in extract_target_sentences(document):
            for morpheme in sentence.morphemes:
                self.global_index2attributes[morpheme.global_index] = (
                    morpheme.pos,
                    morpheme.subpos,
                    morpheme.conjtype,
                    morpheme.conjform,
                )

            for morpheme in sentence.morphemes:
                self.global_index2feature_set[morpheme.global_index] = set()
                for sub_word_feature in SUB_WORD_FEATURES:
                    if sub_word_feature in morpheme.features:
                        self.global_index2feature_set[morpheme.global_index].add(sub_word_feature)
            for base_phrase in sentence.base_phrases:
                self.global_index2feature_set[base_phrase.head.global_index].add("基本句-主辞")
                self.global_index2feature_set[base_phrase.morphemes[-1].global_index].add("基本句-区切")
            for phrase in sentence.phrases:
                self.global_index2feature_set[phrase.morphemes[-1].global_index].add("文節-区切")

            self.named_entities += sentence.named_entities
