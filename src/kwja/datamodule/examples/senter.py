from dataclasses import dataclass
from typing import Dict, Optional

from rhoknp import Document
from transformers import BatchEncoding

from kwja.utils.sub_document import is_target_sentence


class SenterExample:
    def __init__(self, example_id: int, encoding: BatchEncoding) -> None:
        self.example_id = example_id
        self.encoding = encoding
        self.doc_id: Optional[str] = None

        # ---------- sentence segmentation ----------
        self.char_global_index2sent_segmentation_tag: Dict[int, str] = {}

    def load_document(self, document: Document) -> None:
        self.doc_id = document.doc_id
        offset: int = 0
        for sentence in document.sentences:
            if not is_target_sentence(sentence):
                offset += len(sentence.text)
                continue
            for morpheme in sentence.morphemes:
                for char_idx_in_morpheme in range(len(morpheme.text)):
                    self.char_global_index2sent_segmentation_tag[char_idx_in_morpheme + offset] = (
                        "B" if char_idx_in_morpheme == 0 and morpheme.index == 0 else "I"
                    )
                offset += len(morpheme.text)


@dataclass(frozen=True)
class SenterInferenceExample:
    example_id: int
    encoding: BatchEncoding
    doc_id: str
