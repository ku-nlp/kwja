from rhoknp import Document
from rhoknp.props import DepType

from jula.utils.constants import SUB_DOC_PAT


class DependencyExample:
    """A single training/test example for dependency parsing."""

    def __init__(self) -> None:
        self.doc_id: str = ""
        self.dependencies: list[tuple[int, int]] = []  # 形態素単位係り受け
        self.candidates: list[list[int]] = []  # 形態素単位係り先選択候補
        self.dependency_types: list[tuple[int, DepType]] = []  # 形態素単位係り受けラベル

    def load(self, document: Document) -> None:
        self.doc_id = document.doc_id
        if SUB_DOC_PAT.search(self.doc_id) is not None:
            target = document.sentences[-1]
        else:
            target = document

        for morpheme in target.morphemes:
            parent = morpheme.parent
            if parent is not None:
                self.dependencies.append((morpheme.global_index, parent.global_index))
            else:  # 係り先がなければ[ROOT]を指す
                self.dependencies.append((morpheme.global_index, -1))

        if SUB_DOC_PAT.search(self.doc_id) is not None:
            offset = sum(len(sentence.morphemes) for sentence in document.sentences[:-1])
            self.candidates.extend([[] for _ in range(offset)])
        for morpheme in target.morphemes:
            intra_morphemes = morpheme.sentence.morphemes
            self.candidates.append([m.global_index for m in intra_morphemes if m != morpheme])

        self.dependency_types = []
        for base_phrase in target.base_phrases:
            for morpheme in base_phrase.morphemes:
                if morpheme == base_phrase.head:
                    self.dependency_types.append((morpheme.global_index, base_phrase.dep_type))
                else:
                    self.dependency_types.append((morpheme.global_index, DepType.DEPENDENCY))
