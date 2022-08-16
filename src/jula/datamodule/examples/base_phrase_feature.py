import re

from rhoknp import Document

from jula.utils.constants import BASE_PHRASE_FEATURES, SUB_DOC_PAT


class BasePhraseFeatureExample:
    """A single training/test example for base phrase feature prediction."""

    IGNORE_VALUE = re.compile(r"節-(前向き)?機能疑?")

    def __init__(self) -> None:
        self.doc_id: str = ""
        self.heads: list[int] = []  # 基本句単位主辞形態素インデックス
        self.features: list[set[str]] = []  # 基本句単位素性リスト

    def load(self, document: Document) -> None:
        self.doc_id = document.doc_id
        if SUB_DOC_PAT.search(self.doc_id) is not None:
            target = document.sentences[-1]
        else:
            target = document

        target_feature_set = set(BASE_PHRASE_FEATURES)
        for base_phrase in target.base_phrases:
            self.heads.append(base_phrase.head.global_index)
            features = {
                k + (f":{v}" if isinstance(v, str) and self.IGNORE_VALUE.match(k) is None else "")
                for k, v in base_phrase.features.items()
            }
            self.features.append(features & target_feature_set)
