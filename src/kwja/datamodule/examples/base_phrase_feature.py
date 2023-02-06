from typing import Dict, Set

from rhoknp import Document

from kwja.utils.constants import BASE_PHRASE_FEATURES, IGNORE_VALUE_FEATURE_PAT
from kwja.utils.sub_document import extract_target_sentences


class BasePhraseFeatureExample:
    """A single training/test example for base phrase feature prediction."""

    def __init__(self) -> None:
        self.doc_id: str = ""
        # 基本句主辞形態素 global index -> 基本句素性リスト
        self.head_morpheme_global_index2feature_set: Dict[int, Set[str]] = {}

    def load(self, document: Document) -> None:
        self.doc_id = document.doc_id

        target_feature_set = set(BASE_PHRASE_FEATURES)
        for base_phrase in [bp for s in extract_target_sentences(document) for bp in s.base_phrases]:
            features = {
                k + (f":{v}" if isinstance(v, str) and IGNORE_VALUE_FEATURE_PAT.match(k) is None else "")
                for k, v in base_phrase.features.items()
            }
            self.head_morpheme_global_index2feature_set[base_phrase.head.global_index] = features & target_feature_set
