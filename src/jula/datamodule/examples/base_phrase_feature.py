from rhoknp import Document

from jula.utils.constants import BASE_PHRASE_FEATURES


class BasePhraseFeatureExample:
    """A single training/test example for base phrase feature prediction."""

    def __init__(self) -> None:
        self.example_id: int = -1
        self.doc_id: str = ""
        self.heads: list[int] = []  # 基本句単位主辞形態素インデックス
        self.features: list[set[str]] = []  # 基本句単位素性リスト

    def load(self, document: Document) -> None:
        self.doc_id = document.doc_id
        target_feature_set = set(BASE_PHRASE_FEATURES)
        for base_phrase in document.base_phrases:
            self.heads.append(base_phrase.head.global_index)
            features = {k + (f":{v}" if isinstance(v, str) else "") for k, v in base_phrase.features.items()}
            self.features.append(features & target_feature_set)
