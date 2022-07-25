from rhoknp import Document
from rhoknp.units.utils import DepType


class DependencyExample:
    """A single training/test example for dependency parsing."""

    def __init__(self) -> None:
        self.doc_id: str = ""
        self.dependencies: list[int] = []  # 形態素単位係り受け
        self.candidates: list[list[int]] = []  # 形態素単位係り先選択候補
        self.dependency_types: list[DepType] = []  # 形態素単位係り受けラベル

    def load(self, document: Document) -> None:
        self.doc_id = document.doc_id
        for morpheme in document.morphemes:
            parent = morpheme.parent
            if parent is not None:
                self.dependencies.append(parent.global_index)
            else:  # 係り先がなければ[ROOT]を指す
                self.dependencies.append(-1)

        for sentence in document.sentences:
            intra_morphemes = sentence.morphemes
            for morpheme in sentence.morphemes:
                self.candidates.append([m.global_index for m in intra_morphemes if m != morpheme])

        self.dependency_types = [DepType.DEPENDENCY] * len(document.morphemes)
        for base_phrase in document.base_phrases:
            self.dependency_types[base_phrase.head.global_index] = base_phrase.dep_type
