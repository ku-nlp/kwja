from rhoknp import Document

from jula.utils.constants import SEG_TYPES
from jula.utils.sub_document import is_target_sentence


class CharFeatureExample:
    """A single training/test example for char feature prediction."""

    def __init__(self) -> None:
        self.doc_id: str = ""
        self.types: dict[int, int] = {}  # 文字 index -> 単語分割用のBIタグ

    def load(self, document: Document) -> None:
        self.doc_id = document.doc_id
        char_idx: int = 0
        for sentence in document.sentences:
            if is_target_sentence(sentence):
                for morpheme in sentence.morphemes:
                    for char_idx_in_morpheme in range(len(morpheme.text)):
                        self.types[char_idx] = (
                            SEG_TYPES.index("B") if char_idx_in_morpheme == 0 else SEG_TYPES.index("I")
                        )
                        char_idx += 1
            else:
                char_idx += len(sentence.text)
