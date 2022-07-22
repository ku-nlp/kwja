import torch
from rhoknp import Document
from rhoknp.rel import ExophoraReferent
from scipy.special import softmax
from tokenizers import Encoding
from transformers.utils import PaddingStrategy

from jula.datamodule.datasets.base_dataset import BaseDataset
from jula.datamodule.examples import (
    BasePhraseFeatureExample,
    CohesionExample,
    DependencyExample,
    Task,
    WordFeatureExample,
)
from jula.datamodule.extractors import BridgingExtractor, CoreferenceExtractor, PasExtractor
from jula.datamodule.extractors.base import Phrase
from jula.utils.constants import (
    BASE_PHRASE_FEATURES,
    CONJFORM_TYPES,
    CONJTYPE_TYPES,
    DEPENDENCY_TYPE2INDEX,
    DISCOURSE_RELATIONS,
    IGNORE_INDEX,
    POS_TYPES,
    SUBPOS_TYPES,
    WORD_FEATURES,
)


class WordDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        self.exophora_referents: list[ExophoraReferent] = [
            ExophoraReferent(s) for s in kwargs.pop("exophora_referents")
        ]
        self.special_tokens: list[str] = [str(e) for e in self.exophora_referents] + [
            "[NULL]",
            "[NA]",
            "[ROOT]",  # TODO: mask in cohesion analysis
        ]
        tokenizer_kwargs = {
            "additional_special_tokens": self.special_tokens,
        }
        if "tokenizer_kwargs" in kwargs:
            kwargs["tokenizer_kwargs"].update(tokenizer_kwargs)
        else:
            kwargs["tokenizer_kwargs"] = tokenizer_kwargs

        super().__init__(*args, **kwargs)

        self.cohesion_tasks: list[Task] = [Task(t) for t in kwargs["cohesion_tasks"]]
        self.cases: list[str] = list(kwargs["cases"])
        self.pas_targets: list[str] = ["pred", "noun"]
        self.bar_rels = list(kwargs["bar_rels"])
        self.extractors = {
            Task.PAS_ANALYSIS: PasExtractor(self.cases, self.pas_targets, self.exophora_referents, kc=False),
            Task.COREFERENCE: CoreferenceExtractor(self.exophora_referents, kc=False),
            Task.BRIDGING: BridgingExtractor(self.bar_rels, self.exophora_referents, kc=False),
        }
        self.special_to_index: dict[str, int] = {
            token: self.max_seq_length - len(self.special_tokens) + i for i, token in enumerate(self.special_tokens)
        }
        self.word_feature_examples: dict[str, WordFeatureExample] = {}
        self.base_phrase_feature_examples: dict[str, BasePhraseFeatureExample] = {}
        self.dependency_examples: dict[str, DependencyExample] = {}
        self.cohesion_examples: dict[str, CohesionExample] = {}
        for example_id, document in enumerate(self.documents):
            word_feature_example = WordFeatureExample()
            word_feature_example.load(document)
            word_feature_example.example_id = example_id
            self.word_feature_examples[document.doc_id] = word_feature_example

            base_phrase_feature_example = BasePhraseFeatureExample()
            base_phrase_feature_example.load(document)
            base_phrase_feature_example.example_id = example_id
            self.base_phrase_feature_examples[document.doc_id] = base_phrase_feature_example

            dependency_example = DependencyExample()
            dependency_example.load(document)
            dependency_example.example_id = example_id
            self.dependency_examples[document.doc_id] = dependency_example

            cohesion_example = CohesionExample()
            cohesion_example.load(document, tasks=self.cohesion_tasks, extractors=self.extractors)
            cohesion_example.example_id = example_id
            self.cohesion_examples[document.doc_id] = cohesion_example

    @property
    def special_indices(self) -> list[int]:
        return list(self.special_to_index.values())

    @property
    def num_special_tokens(self) -> int:
        return len(self.special_tokens)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        document = self.documents[index]
        word_feature_example = self.word_feature_examples[document.doc_id]
        base_phrase_feature_example = self.base_phrase_feature_examples[document.doc_id]
        dependency_example = self.dependency_examples[document.doc_id]
        cohesion_example = self.cohesion_examples[document.doc_id]
        return self.encode(
            document, word_feature_example, base_phrase_feature_example, dependency_example, cohesion_example
        )

    def encode(
        self,
        document: Document,
        word_feature_example: WordFeatureExample,
        base_phrase_feature_example: BasePhraseFeatureExample,
        dependency_example: DependencyExample,
        cohesion_example: CohesionExample,
    ) -> dict[str, torch.Tensor]:
        # TODO: deal with the case that the document is too long
        text = " ".join(morpheme.text for morpheme in document.morphemes)

        encoding: Encoding = self.tokenizer(
            text,
            truncation=True,
            padding=PaddingStrategy.MAX_LENGTH,
            max_length=self.max_seq_length - self.num_special_tokens,
        ).encodings[0]

        # NOTE: hereafter, indices are given at the word level
        mrph_types: list[list[int]] = [[IGNORE_INDEX] * 4 for _ in range(self.max_seq_length)]
        for morpheme in document.morphemes:
            if morpheme.pos in POS_TYPES:
                mrph_types[morpheme.global_index][0] = POS_TYPES.index(morpheme.pos)
            if morpheme.subpos in SUBPOS_TYPES:
                mrph_types[morpheme.global_index][1] = SUBPOS_TYPES.index(morpheme.subpos)
            if morpheme.conjtype in CONJTYPE_TYPES:
                mrph_types[morpheme.global_index][2] = CONJTYPE_TYPES.index(morpheme.conjtype)
            if morpheme.conjtype in CONJTYPE_TYPES:
                mrph_types[morpheme.global_index][3] = CONJFORM_TYPES.index(morpheme.conjform)

        # word feature tagging
        word_features = [[IGNORE_INDEX] * len(WORD_FEATURES) for _ in range(self.max_seq_length)]
        for morpheme_index, feature_set in enumerate(word_feature_example.features):
            for i, word_feature in enumerate(WORD_FEATURES):
                word_features[morpheme_index][i] = int(word_feature in feature_set)

        # base phrase feature tagging
        base_phrase_features = [[IGNORE_INDEX] * len(BASE_PHRASE_FEATURES) for _ in range(self.max_seq_length)]
        for head_index, feature_set in zip(base_phrase_feature_example.heads, base_phrase_feature_example.features):
            for i, base_phrase_feature in enumerate(BASE_PHRASE_FEATURES):
                base_phrase_features[head_index][i] = int(base_phrase_feature in feature_set)

        # dependency parsing
        dependencies: list[int] = [IGNORE_INDEX for _ in range(self.max_seq_length)]
        for global_morpheme_index, dependency in enumerate(dependency_example.dependencies):
            dependencies[global_morpheme_index] = dependency if dependency != -1 else self.special_to_index["[ROOT]"]

        dependency_mask: list[list[bool]] = []  # False -> mask, True -> keep
        for cands in dependency_example.candidates:
            cands.append(self.special_to_index["[ROOT]"])
            dependency_mask.append([(x in cands) for x in range(self.max_seq_length)])
        dependency_mask += [[False] * self.max_seq_length] * (self.max_seq_length - len(dependency_mask))  # pad

        dependency_types: list[int] = [IGNORE_INDEX for _ in range(self.max_seq_length)]
        for global_morpheme_index, dependency_type in enumerate(dependency_example.dependency_types):
            dependency_types[global_morpheme_index] = DEPENDENCY_TYPE2INDEX[dependency_type]

        # PAS analysis & coreference resolution
        cohesion_example.encoding = encoding
        cohesion_target: list[list[list[int]]] = []  # (task, src, tgt)
        candidates_set: list[list[list[int]]] = []  # (task, src, tgt)
        if Task.PAS_ANALYSIS in self.cohesion_tasks:
            task = Task.PAS_ANALYSIS
            annotation = cohesion_example.annotations[task]
            phrases = cohesion_example.phrases[task]
            for case in self.cases:
                arguments_set = [arguments[case] for arguments in annotation.arguments_set]
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
        cohesion_mask = [
            [[(x in cands) for x in range(self.max_seq_length)] for cands in candidates]
            for candidates in candidates_set
        ]  # False -> mask, True -> keep

        # TODO: discourse relation analysis
        discourse_relations = [
            [[DISCOURSE_RELATIONS.index("談話関係なし")] * len(DISCOURSE_RELATIONS) for _ in range(self.max_seq_length)]
            for _ in range(self.max_seq_length)
        ]

        special_encoding: Encoding = self.tokenizer(
            self.special_tokens,
            is_split_into_words=True,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=False,
            add_special_tokens=False,
        ).encodings[0]
        merged_encoding: Encoding = Encoding.merge([encoding, special_encoding])

        return {
            "example_ids": torch.tensor(cohesion_example.example_id, dtype=torch.long),
            "input_ids": torch.tensor(merged_encoding.ids, dtype=torch.long),
            "attention_mask": torch.tensor(merged_encoding.attention_mask, dtype=torch.long),
            "subword_map": torch.tensor(self._gen_subword_map(encoding), dtype=torch.bool),
            "mrph_types": torch.tensor(mrph_types, dtype=torch.long),
            "word_features": torch.tensor(word_features, dtype=torch.long),
            "base_phrase_features": torch.tensor(base_phrase_features, dtype=torch.long),
            "dependencies": torch.tensor(dependencies, dtype=torch.long),
            "intra_mask": torch.tensor(dependency_mask, dtype=torch.bool),
            "dependency_types": torch.tensor(dependency_types, dtype=torch.long),
            "discourse_relations": torch.tensor(discourse_relations, dtype=torch.long),
            "cohesion_target": torch.tensor(cohesion_target, dtype=torch.int),
            "cohesion_mask": torch.tensor(cohesion_mask, dtype=torch.bool),
            "texts": text,
        }

    def dump_prediction(
        self,
        result: list[list[list[float]]],  # word level
        example: CohesionExample,
    ) -> list[list[list[float]]]:  # (phrase, rel, 0 or phrase+special)
        """1 example 中に存在する基本句それぞれに対してシステム予測のリストを返す．"""
        ret: list[list[list[float]]] = [[] for _ in next(iter(example.phrases.values()))]
        task_idx = 0
        if Task.PAS_ANALYSIS in self.cohesion_tasks:
            for _ in self.cases:
                for i, p in enumerate(self._token2bp_level(result[task_idx], example.phrases[Task.PAS_ANALYSIS])):
                    ret[i].append(p)
                task_idx += 1
        if Task.BRIDGING in self.cohesion_tasks:
            for i, p in enumerate(self._token2bp_level(result[task_idx], example.phrases[Task.BRIDGING])):
                ret[i].append(p)
            task_idx += 1
        if Task.COREFERENCE in self.cohesion_tasks:
            for i, p in enumerate(self._token2bp_level(result[task_idx], example.phrases[Task.COREFERENCE])):
                ret[i].append(p)
            task_idx += 1
        return ret

    def _token2bp_level(
        self,
        word_level_output: list[list[float]],
        phrases: list[Phrase],
    ) -> list[list[float]]:  # (bp, 0 or bp+special)
        ret: list[list[float]] = [[] for _ in phrases]
        for anaphor in filter(lambda p: p.is_target, phrases):
            word_level_scores: list[float] = word_level_output[anaphor.dmid]
            phrase_level_scores: list[float] = []
            for tgt_bp in phrases:
                if tgt_bp.dtid not in anaphor.candidates:
                    phrase_level_scores.append(-1)  # pad -1 for non-candidate phrases
                    continue
                phrase_level_scores.append(word_level_scores[tgt_bp.dmid])
            phrase_level_scores += [word_level_scores[idx] for idx in self.special_indices]
            assert len(phrase_level_scores) == len(phrases) + len(self.special_to_index)
            ret[anaphor.dtid] = softmax(phrase_level_scores).tolist()
        return ret

    def _gen_subword_map(self, encoding: Encoding) -> list[list[bool]]:
        subword_map = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        for token_id, word_id in enumerate(encoding.word_ids):
            if word_id is not None:
                subword_map[word_id][token_id] = True
        return subword_map

    def _convert_annotation_to_feature(
        self,
        annotation: list[list[str]],  # phrase level
        phrases: list[Phrase],
    ) -> tuple[list[list[int]], list[list[int]]]:
        scores_set: list[list[int]] = [[0] * self.max_seq_length for _ in range(self.max_seq_length)]
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
                    else:
                        word_index = phrases[int(arg_string)].dmid
                    scores[word_index] = 1

                word_level_candidates: list[int] = []
                for candidate in candidates:
                    word_level_candidates.append(phrases[candidate].dmid)
                word_level_candidates += self.special_indices

                # use the head subword as the representative of the source word
                scores_set[mrph.dmid] = scores
                candidates_set[mrph.dmid] = word_level_candidates

        return scores_set, candidates_set  # word level
