from rhoknp import Document
from rhoknp.props import DepType

from kwja.utils.sub_document import extract_target_sentences


class DependencyExample:
    """A single training/test example for dependency parsing."""

    def __init__(self) -> None:
        self.doc_id: str = ""
        self.dependencies: dict[int, int] = {}  # 形態素単位係り受け
        self.candidates: dict[int, list[int]] = {}  # 形態素単位係り先選択候補
        self.dependency_types: dict[int, DepType] = {}  # 形態素単位係り受けラベル

    def load(self, document: Document) -> None:
        self.doc_id = document.doc_id
        for sentence in extract_target_sentences(document):
            for morpheme in sentence.morphemes:
                parent = morpheme.parent
                # 係り先がなければ-1
                self.dependencies[morpheme.global_index] = parent.global_index if parent is not None else -1

            intra_morphemes = sentence.morphemes
            for morpheme in intra_morphemes:
                self.candidates[morpheme.global_index] = [m.global_index for m in intra_morphemes if m != morpheme]

            for base_phrase in sentence.base_phrases:
                for morpheme in base_phrase.morphemes:
                    if morpheme == base_phrase.head:
                        dependency_type = base_phrase.dep_type
                    else:
                        dependency_type = DepType.DEPENDENCY
                    self.dependency_types[morpheme.global_index] = dependency_type
