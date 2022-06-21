import torch
from rhoknp import Document
from rhoknp.rel import ExophoraReferent
from tokenizers import Encoding
from transformers.file_utils import PaddingStrategy

from jula.datamodule.datasets.base_dataset import BaseDataset
from jula.datamodule.examples import CohesionExample, Task
from jula.datamodule.extractors import (
    BridgingExtractor,
    CoreferenceExtractor,
    PasExtractor,
)
from jula.datamodule.extractors.base import Phrase
from jula.utils.utils import (
    BASE_PHRASE_FEATURES,
    DEPENDENCY_TYPES,
    DISCOURSE_RELATIONS,
    IGNORE_INDEX,
    WORD_FEATURES,
)


class WordDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        self.exophora_referents: list[ExophoraReferent] = [
            ExophoraReferent(s) for s in ("著者", "読者", "不特定:人", "不特定:物")
        ]
        self.special_tokens: list[str] = [str(e) for e in self.exophora_referents] + [
            "NULL",
            "NA",
            "[ROOT]",
        ]
        tokenizer_kwargs = {
            "additional_special_tokens": self.special_tokens,
        }
        if "tokenizer_kwargs" in kwargs:
            kwargs["tokenizer_kwargs"].update(tokenizer_kwargs)
        else:
            kwargs["tokenizer_kwargs"] = tokenizer_kwargs

        super().__init__(*args, **kwargs)

        self.cohesion_tasks: list[Task] = [
            Task.PAS_ANALYSIS,
            Task.BRIDGING,
            Task.COREFERENCE,
        ]
        self.cases = ["ガ", "ヲ", "ニ", "ガ２"]
        self.pas_targets: list[str] = ["pred", "noun"]
        self.bar_rels = ["ノ", "ノ？"]
        self.extractors = {
            Task.PAS_ANALYSIS: PasExtractor(
                self.cases, self.pas_targets, self.exophora_referents, kc=False
            ),
            Task.COREFERENCE: CoreferenceExtractor(self.exophora_referents, kc=False),
            Task.BRIDGING: BridgingExtractor(
                self.bar_rels, self.exophora_referents, kc=False
            ),
        }
        self.special_to_index: dict[str, int] = {
            token: self.max_seq_length - len(self.special_tokens) + i
            for i, token in enumerate(self.special_tokens)
        }
        self.did2example = {}

    @property
    def special_indices(self) -> list[int]:
        return list(self.special_to_index.values())

    @property
    def num_special_tokens(self) -> int:
        return len(self.special_tokens)

    def _load_cohesion_example(self, document: Document) -> CohesionExample:
        example = CohesionExample()
        example.load(document, tasks=self.cohesion_tasks, extractors=self.extractors)
        return example

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        document = self.documents[index]
        return {
            "document_id": torch.tensor(index, dtype=torch.long),
            **self.encode(document),
        }

    def encode(self, document: Document) -> dict[str, torch.Tensor]:
        # TODO: deal with the case that the document is too long
        encoding: Encoding = self.tokenizer(
            " ".join(morpheme.text for morpheme in document.morphemes),
            truncation=True,
            padding=PaddingStrategy.MAX_LENGTH,
            max_length=self.max_seq_length - self.num_special_tokens,
        ).encodings[0]

        # NOTE: hereafter, indices are given at the word level
        word_features = [[0] * len(WORD_FEATURES) for _ in range(self.max_seq_length)]
        for base_phrase in document.base_phrases:
            head = base_phrase.head
            end = base_phrase.morphemes[-1]
            word_features[head.global_index][WORD_FEATURES.index("基本句-主辞")] = 1
            word_features[end.global_index][WORD_FEATURES.index("基本句-区切")] = 1
        for phrase in document.phrases:
            end = phrase.morphemes[-1]
            word_features[end.global_index][WORD_FEATURES.index("文節-区切")] = 1

        base_phrase_features = [
            [IGNORE_INDEX] * len(BASE_PHRASE_FEATURES)
            for _ in range(self.max_seq_length)
        ]
        for base_phrase in document.base_phrases:
            for i, base_phrase_feature in enumerate(BASE_PHRASE_FEATURES):
                if ":" in base_phrase_feature:
                    key, value = base_phrase_feature.split(":")
                else:
                    key, value = base_phrase_feature, ""
                head = base_phrase.head
                if base_phrase.features.get(key, False) in (value, True):
                    base_phrase_features[head.global_index][i] = 1
                else:
                    base_phrase_features[head.global_index][i] = 0

        dependencies = [IGNORE_INDEX for _ in range(self.max_seq_length)]
        for morpheme in document.morphemes:
            parent = morpheme.parent
            if parent:
                dependencies[morpheme.global_index] = morpheme.parent.global_index
            else:  # 係り先がなければ文末の[ROOT]を指す
                dependencies[morpheme.global_index] = self.special_to_index["[ROOT]"]

        intra_mask = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        for sentence in document.sentences:
            morpheme_global_indices = [
                morpheme.global_index for morpheme in sentence.morphemes
            ]
            begin, end = map(lambda x: x(morpheme_global_indices), [min, max])
            for i in range(begin, end + 1):
                # 末尾の基本句主辞だけ[ROOT]を指す (複数の係り先が[ROOT]を指さないように)
                if i == sentence.base_phrases[-1].head.global_index:
                    intra_mask[i][-1] = True
                else:
                    for j in range(begin, end + 1):
                        if i != j:
                            intra_mask[i][j] = True

        dependency_types = [IGNORE_INDEX for _ in range(self.max_seq_length)]
        for base_phrase in document.base_phrases:
            for morpheme in base_phrase.morphemes:
                if base_phrase.head == morpheme:
                    dependency_types[morpheme.global_index] = DEPENDENCY_TYPES.index(
                        base_phrase.dep_type.value
                    )
                else:
                    dependency_types[morpheme.global_index] = DEPENDENCY_TYPES.index(
                        "D"
                    )

        # PAS analysis & coreference resolution
        cohesion_example = self._load_cohesion_example(document)
        self.did2example[document.doc_id] = cohesion_example
        cohesion_example.encoding = encoding
        cohesion_target: list[list[list[int]]] = []  # (task, src, tgt)
        candidates_set: list[list[list[int]]] = []  # (task, src, tgt)
        if Task.PAS_ANALYSIS in self.cohesion_tasks:
            task = Task.PAS_ANALYSIS
            annotation = cohesion_example.annotations[task]
            phrases = cohesion_example.phrases[task]
            for case in self.cases:
                arguments_set = [
                    arguments[case] for arguments in annotation.arguments_set
                ]
                ret = self._convert_annotation_to_feature(arguments_set, phrases)
                cohesion_target.append(ret[0])
                candidates_set.append(ret[1])
        for task in (Task.BRIDGING, Task.COREFERENCE):
            if task not in self.cohesion_tasks:
                continue
            annotation = cohesion_example.annotations[task].arguments_set
            phrases = cohesion_example.phrases[task]
            ret = self._convert_annotation_to_feature(annotation, phrases)
            cohesion_target.append(ret[0])
            candidates_set.append(ret[1])

        special_encoding: Encoding = self.tokenizer(
            self.special_tokens,
            is_split_into_words=True,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=False,
            add_special_tokens=False,
        ).encodings[0]
        merged_encoding: Encoding = Encoding.merge([encoding, special_encoding])
        cohesion_mask = [
            [[(x in cands) for x in range(self.max_seq_length)] for cands in candidates]
            for candidates in candidates_set
        ]  # False -> mask, True -> keep

        # TODO: discourse relation analysis
        discourse_relations = [
            [
                [DISCOURSE_RELATIONS.index("談話関係なし")] * len(DISCOURSE_RELATIONS)
                for _ in range(self.max_seq_length)
            ]
            for _ in range(self.max_seq_length)
        ]

        return {
            "input_ids": torch.tensor(merged_encoding.ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                merged_encoding.attention_mask, dtype=torch.long
            ),
            "subword_map": torch.tensor(
                self._gen_subword_map(encoding), dtype=torch.bool
            ),
            "word_features": torch.tensor(word_features, dtype=torch.float),
            "base_phrase_features": torch.tensor(
                base_phrase_features, dtype=torch.float
            ),
            "num_base_phrases": torch.tensor(
                len(document.base_phrases), dtype=torch.long
            ),
            "dependencies": torch.tensor(dependencies, dtype=torch.long),
            "intra_mask": torch.tensor(intra_mask, dtype=torch.bool),
            "dependency_types": torch.tensor(dependency_types, dtype=torch.long),
            "discourse_relations": torch.tensor(discourse_relations, dtype=torch.long),
            "cohesion_target": torch.tensor(cohesion_target, dtype=torch.int),
            "cohesion_mask": torch.tensor(cohesion_mask, dtype=torch.bool),
        }

    def _gen_subword_map(self, encoding: Encoding) -> list[list[bool]]:
        subword_map = [
            [False] * self.max_seq_length for _ in range(self.max_seq_length)
        ]
        for token_id, word_id in enumerate(encoding.word_ids):
            if word_id is not None:
                subword_map[word_id][token_id] = True
        return subword_map

    def _convert_annotation_to_feature(
        self,
        annotation: list[list[str]],  # phrase level
        phrases: list[Phrase],
    ) -> tuple[list[list[int]], list[list[int]]]:
        scores_set: list[list[int]] = [
            [False] * self.max_seq_length for _ in range(self.max_seq_length)
        ]
        candidates_set: list[list[int]] = [[] for _ in range(self.max_seq_length)]

        for phrase in phrases:
            arguments: list[str] = annotation[phrase.dtid]
            candidates: list[int] = phrase.candidates  # phrase level
            for mrph in filter(lambda m: m.is_target, phrase.children):
                scores: list[int] = [0] * self.max_seq_length
                for arg_string in arguments:
                    # arg_string: 著者, 8%C, 15%O, 2, NULL, ...
                    if arg_string[-2:] in ("%C", "%N", "%O"):
                        # PAS only
                        # flag = arg_string[-1]
                        arg_string = arg_string[:-2]
                    if arg_string in self.special_to_index:
                        word_index = self.special_to_index[arg_string]
                        scores[word_index] = 1
                    else:
                        scores[phrases[int(arg_string)].dmid] = 1

                word_level_candidates: list[int] = []
                for candidate in candidates:
                    word_level_candidates.append(phrases[candidate].dmid)
                word_level_candidates += self.special_indices

                # use the head subword as the representative of the source word
                scores_set[mrph.dmid] = scores
                candidates_set[mrph.dmid] = word_level_candidates

        return scores_set, candidates_set  # word level
