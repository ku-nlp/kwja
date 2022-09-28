from rhoknp import Document

from kwja.utils.constants import BASE_PHRASE_FEATURES, IGNORE_VALUE_FEATURE_PAT
from kwja.utils.sub_document import extract_target_sentences


class BasePhraseFeatureExample:
    """A single training/test example for base phrase feature prediction."""

    def __init__(self) -> None:
        self.doc_id: str = ""
        self.heads: list[int] = []  # 基本句単位主辞形態素インデックス
        self.features: list[set[str]] = []  # 基本句単位素性リスト

    def load(self, document: Document) -> None:
        self.doc_id = document.doc_id

        target_feature_set = set(BASE_PHRASE_FEATURES)
        for base_phrase in [bp for sent in extract_target_sentences(document) for bp in sent.base_phrases]:
            self.heads.append(base_phrase.head.global_index)
            features = {
                k + (f":{v}" if isinstance(v, str) and IGNORE_VALUE_FEATURE_PAT.match(k) is None else "")
                for k, v in base_phrase.features.items()
            }
            self.features.append(features & target_feature_set)
