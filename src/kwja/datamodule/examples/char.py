from dataclasses import dataclass
from typing import Dict, Optional

from rhoknp import Document, Morpheme
from transformers import BatchEncoding

from kwja.utils.sub_document import is_target_sentence
from kwja.utils.word_normalization import MorphemeNormalizer


class CharExample:
    def __init__(self, example_id: int, encoding: BatchEncoding) -> None:
        self.example_id = example_id
        self.encoding = encoding
        self.doc_id: Optional[str] = None

        # ---------- sentence segmentation ----------
        self.char_global_index2sent_segmentation_tag: Dict[int, str] = {}

        # ---------- word segmentation ----------
        self.char_global_index2word_segmentation_tag: Dict[int, str] = {}

        # ---------- word normalization ----------
        self.char_global_index2word_norm_op_tag: Dict[int, str] = {}

        self.normalizer: MorphemeNormalizer = MorphemeNormalizer()

    def load_document(self, document: Document) -> None:
        self.doc_id = document.doc_id
        offset: int = 0
        for sentence in document.sentences:
            if not is_target_sentence(sentence):
                offset += len(sentence.text)
                continue
            for morpheme in sentence.morphemes:
                self._set_sent_segmentation_tag(morpheme, offset)
                self._set_word_segmentation_tag(morpheme, offset)
                self._set_word_norm_op_tag(morpheme, offset)
                offset += len(morpheme.text)

    def _set_sent_segmentation_tag(self, morpheme: Morpheme, offset: int) -> None:
        for char_index_in_morpheme in range(len(morpheme.text)):
            self.char_global_index2sent_segmentation_tag[char_index_in_morpheme + offset] = (
                "B" if char_index_in_morpheme == 0 and morpheme.index == 0 else "I"
            )

    def _set_word_segmentation_tag(self, morpheme: Morpheme, offset: int) -> None:
        for char_index_in_morpheme in range(len(morpheme.text)):
            self.char_global_index2word_segmentation_tag[char_index_in_morpheme + offset] = (
                "B" if char_index_in_morpheme == 0 else "I"
            )

    def _set_word_norm_op_tag(self, morpheme: Morpheme, offset: int) -> None:
        for char_index_in_morpheme, word_norm_op_tag in enumerate(self.normalizer.get_word_norm_op_tags(morpheme)):
            self.char_global_index2word_norm_op_tag[char_index_in_morpheme + offset] = word_norm_op_tag


@dataclass(frozen=True)
class CharInferenceExample:
    example_id: int
    encoding: BatchEncoding
    doc_id: str
