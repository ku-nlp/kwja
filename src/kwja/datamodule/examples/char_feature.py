from rhoknp import Document

from kwja.utils.constants import IGNORE_INDEX, IGNORE_WORD_NORM_TYPE, SEG_TYPES, WORD_NORM_TYPES
from kwja.utils.sub_document import is_target_sentence
from kwja.utils.word_normalize import MorphemeNormalizer


class CharFeatureExample:
    """A single training/test example for char feature prediction."""

    def __init__(self) -> None:
        self.doc_id: str = ""
        self.seg_types: dict[int, int] = {}  # 文字 index -> 単語分割用のBIタグ
        self.norm_types: dict[int, int] = {}  # 文字 index -> 正規化タイプ

        self.normalizer: MorphemeNormalizer = MorphemeNormalizer()

    def load(self, document: Document) -> None:
        self.doc_id = document.doc_id
        offset: int = 0
        for sentence in document.sentences:
            if not is_target_sentence(sentence):
                offset += len(sentence.text)
                continue
            for morpheme in sentence.morphemes:
                for char_idx_in_morpheme in range(len(morpheme.text)):
                    self.seg_types[offset + char_idx_in_morpheme] = (
                        SEG_TYPES.index("B") if char_idx_in_morpheme == 0 else SEG_TYPES.index("I")
                    )
                for char_idx_in_morpheme, opn in enumerate(self.normalizer.get_normalization_opns(morpheme)):
                    if opn == IGNORE_WORD_NORM_TYPE:
                        self.norm_types[offset + char_idx_in_morpheme] = IGNORE_INDEX
                    else:
                        self.norm_types[offset + char_idx_in_morpheme] = WORD_NORM_TYPES.index(opn)
                offset += len(morpheme.text)
