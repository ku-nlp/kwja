import torch
from rhoknp import Document
from transformers import BatchEncoding

from jula.datamodule.datasets.base_dataset import BaseDataset
from jula.utils.utils import (
    BASE_PHRASE_FEATURES,
    DEPENDENCY_TYPES,
    DISCOURSE_RELATIONS,
    IGNORE_INDEX,
    WORD_FEATURES,
)


class WordDataset(BaseDataset):
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        document = self.documents[index]
        return {
            "document_id": torch.tensor(index, dtype=torch.long),
            **self.encode(document),
        }

    def encode(self, document: Document) -> dict[str, torch.Tensor]:
        # TODO: deal with the case that the document is too long
        encoding: BatchEncoding = self.tokenizer(
            " ".join(morpheme.text for morpheme in document.morphemes),
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length - 1,
        )
        input_ids = encoding["input_ids"] + [self.tokenizer.vocab["[ROOT]"]]
        attention_mask = encoding["attention_mask"] + [1]
        subword_map = [
            [False] * self.max_seq_length for _ in range(self.max_seq_length)
        ]
        for token_id, word_id in enumerate(encoding.word_ids()):
            if word_id is not None:
                subword_map[word_id][token_id] = True

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
                dependencies[morpheme.global_index] = self.max_seq_length - 1

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

        # TODO: PAS analysis & coreference resolution
        # TODO: discourse relation analysis
        discourse_relations = [
            [
                [DISCOURSE_RELATIONS.index("談話関係なし")] * len(DISCOURSE_RELATIONS)
                for _ in range(self.max_seq_length)
            ]
            for _ in range(self.max_seq_length)
        ]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "subword_map": torch.tensor(subword_map, dtype=torch.bool),
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
        }
