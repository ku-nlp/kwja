from typing import Dict

from rhoknp import Document

from kwja.utils.sub_document import is_target_sentence
from kwja.utils.word_normalization import MorphemeNormalizer


class CharFeatureExample:
    """A single training/test example for char feature prediction."""

    def __init__(self) -> None:
        self.doc_id: str = ""
        self.index2word_segmentation_tag: Dict[int, str] = {}  # 文字 index -> 単語分割タグ
        self.index2word_norm_op_tag: Dict[int, str] = {}  # 文字 index -> 単語正規化操作タグ

        self.normalizer: MorphemeNormalizer = MorphemeNormalizer()

    def load(self, document: Document) -> None:
        self.doc_id = document.doc_id
        offset: int = 0
        for sentence in document.sentences:
            if not is_target_sentence(sentence):
                offset += len(sentence.text)
                continue
            for morpheme in sentence.morphemes:
                for char_index_in_morpheme in range(len(morpheme.text)):
                    self.index2word_segmentation_tag[char_index_in_morpheme + offset] = (
                        "B" if char_index_in_morpheme == 0 else "I"
                    )
                for char_index_in_morpheme, word_norm_op_tag in enumerate(
                    self.normalizer.get_word_norm_op_tags(morpheme)
                ):
                    self.index2word_norm_op_tag[char_index_in_morpheme + offset] = word_norm_op_tag
                offset += len(morpheme.text)
