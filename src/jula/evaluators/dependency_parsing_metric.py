from typing import Union

import torch
from rhoknp import BasePhrase, Document, Morpheme, Sentence
from torchmetrics import Metric

from jula.datamodule.datasets.word_dataset import WordDataset
from jula.evaluators.my_conll18_ud_eval import main as conll18_ud_eval
from jula.utils.utils import INDEX2DEPENDENCY_TYPE


class DependencyParsingMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("example_ids", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("type_preds", default=[], dist_reduce_fx="cat")

    def update(
        self, example_ids: torch.Tensor, preds: torch.Tensor, type_preds: torch.Tensor
    ) -> None:
        self.example_ids.append(example_ids)
        self.preds.append(preds)
        self.type_preds.append(type_preds)

    def compute(self, dataset: WordDataset) -> dict[str, Union[torch.Tensor, float]]:
        sorted_indices = self.unique(self.example_ids)
        example_ids, preds, type_preds = map(
            lambda x: x[sorted_indices].tolist(),
            (self.example_ids, self.preds, self.type_preds),
        )
        documents = [dataset.documents[example_id] for example_id in example_ids]

        base_phrase_based_metrics = conll18_ud_eval(
            *self._to_base_phrase_based_conll(documents, preds, type_preds)
        )
        morpheme_based_metrics = conll18_ud_eval(
            *self._to_morpheme_based_conll(documents, preds, type_preds)
        )
        metrics = {
            "base_phrase_" + key: value
            for key, value in base_phrase_based_metrics.items()
        }
        metrics.update(
            {"morpheme_" + key: value for key, value in morpheme_based_metrics.items()}
        )
        return metrics

    @staticmethod
    def unique(x, dim=None):
        unique, inverse = torch.unique(x, sorted=True, return_inverse=True, dim=dim)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        return inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)

    def _to_base_phrase_based_conll(
        self,
        documents: list[Document],
        preds: list[list[int]],
        type_preds: list[list[int]],
    ) -> tuple[list[str], list[str]]:
        gold_lines, system_lines = [], []
        for document, pred, type_pred in zip(documents, preds, type_preds):
            for sentence in document.sentences:
                begin, end, morpheme_global_index2base_phrase_index = self._prefetch(
                    sentence, len(pred)
                )
                for base_phrase in sentence.base_phrases:
                    if base_phrase.parent_index >= 0:
                        gold_head = base_phrase.parent.index + 1
                        gold_deprel = base_phrase.dep_type.value
                    else:
                        gold_head = 0
                        gold_deprel = "ROOT"
                    gold_lines.append(
                        self._to_conll_line(base_phrase, gold_head, gold_deprel)
                    )

                    p, tp = map(
                        lambda x: x[base_phrase.head.global_index], [pred, type_pred]
                    )
                    system_head = morpheme_global_index2base_phrase_index[p]
                    system_deprel = (
                        INDEX2DEPENDENCY_TYPE[tp] if system_head > 0 else "ROOT"
                    )
                    if 0 < system_head <= base_phrase.index:
                        system_head, system_deprel = self._find_another_governor(
                            base_phrase,
                            morpheme_global_index2base_phrase_index,
                            pred,
                            type_pred,
                        )

                    system_lines.append(
                        self._to_conll_line(base_phrase, system_head, system_deprel)
                    )

                gold_lines.append("\n")
                system_lines.append("\n")
        return gold_lines, system_lines

    def _to_morpheme_based_conll(
        self,
        documents: list[Document],
        preds: list[list[int]],
        type_preds: list[list[int]],
    ) -> tuple[list[str], list[str]]:
        gold_lines, system_lines = [], []
        for document, pred, type_pred in zip(documents, preds, type_preds):
            for sentence in document.sentences:
                begin, end, morpheme_global_index2base_phrase_index = self._prefetch(
                    sentence, len(pred)
                )
                for morpheme in sentence.morphemes:
                    if morpheme.parent:
                        gold_head = morpheme.parent.index + 1
                        if morpheme == morpheme.base_phrase.head:
                            gold_deprel = morpheme.base_phrase.dep_type.value
                        else:
                            gold_deprel = "D"
                    else:
                        gold_head = 0
                        gold_deprel = "ROOT"
                    gold_lines.append(
                        self._to_conll_line(morpheme, gold_head, gold_deprel)
                    )

                    p, tp = map(lambda x: x[morpheme.global_index], [pred, type_pred])
                    if begin <= p <= end:
                        system_head = p - begin + 1
                        system_deprel = INDEX2DEPENDENCY_TYPE[tp]
                    else:
                        system_head = 0
                        system_deprel = "ROOT"
                    system_lines.append(
                        self._to_conll_line(morpheme, system_head, system_deprel)
                    )

                gold_lines.append("\n")
                system_lines.append("\n")
        return gold_lines, system_lines

    @staticmethod
    def _prefetch(
        sentence: Sentence, max_seq_len: int
    ) -> tuple[int, int, dict[int, int]]:
        morpheme_global_indices = [
            morpheme.global_index for morpheme in sentence.morphemes
        ]
        begin, end = map(lambda x: x(morpheme_global_indices), [min, max])
        morpheme_global_index2base_phrase_index = {
            morpheme.global_index: base_phrase.index + 1
            for base_phrase in sentence.base_phrases
            for morpheme in base_phrase.morphemes
        }
        morpheme_global_index2base_phrase_index[max_seq_len - 1] = 0
        return begin, end, morpheme_global_index2base_phrase_index

    @staticmethod
    def _to_conll_line(
        unit: Union[BasePhrase, Morpheme], head: int, deprel: str
    ) -> str:
        id_ = unit.index + 1  # 0-origin -> 1-origin
        if isinstance(unit, BasePhrase):
            form = "".join(morpheme.surf for morpheme in unit.morphemes)
            lemma = "".join(morpheme.lemma for morpheme in unit.morphemes)
        else:
            form = unit.surf
            lemma = unit.lemma
        upos = "_"
        xpos = "_"
        feats = "_"
        deps = "_"
        misc = "_"
        return "\t".join(
            map(str, [id_, form, lemma, upos, xpos, feats, head, deprel, deps, misc])
        )

    @staticmethod
    def _find_another_governor(
        base_phrase: BasePhrase,
        morpheme_global_index2base_phrase_index: dict[int, int],
        pred: list[int],
        type_pred,
    ) -> tuple[int, str]:
        for morpheme in base_phrase.morphemes:
            p, tp = map(lambda x: x[morpheme.global_index], [pred, type_pred])
            candidate = morpheme_global_index2base_phrase_index[p]
            if candidate > base_phrase.index:
                return candidate, INDEX2DEPENDENCY_TYPE[tp]
        else:
            return 0, "ROOT"
