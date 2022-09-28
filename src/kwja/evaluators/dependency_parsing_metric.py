from typing import Union

import torch
from rhoknp import BasePhrase, Document, Morpheme
from rhoknp.props import DepType
from torchmetrics import Metric

from kwja.evaluators.conll18_ud_eval import main as conll18_ud_eval
from kwja.utils.constants import INDEX2DEPENDENCY_TYPE
from kwja.utils.dependency_parsing import DependencyManager
from kwja.utils.sub_document import extract_target_sentences


class DependencyParsingMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("example_ids", default=[], dist_reduce_fx="cat")
        self.add_state("dependency_predictions", default=[], dist_reduce_fx="cat")
        self.add_state("dependency_type_predictions", default=[], dist_reduce_fx="cat")

    def update(
        self,
        example_ids: torch.Tensor,
        dependency_predictions: torch.Tensor,
        dependency_type_predictions: torch.Tensor,
    ) -> None:
        self.example_ids.append(example_ids)
        self.dependency_predictions.append(dependency_predictions)
        self.dependency_type_predictions.append(dependency_type_predictions)

    def compute(self, documents: list[Document]) -> dict[str, Union[torch.Tensor, float]]:
        sorted_indices = self.unique(self.example_ids)
        (example_ids, dependency_predictions, dependency_type_predictions) = map(
            lambda x: x[sorted_indices].tolist(),
            [
                self.example_ids,
                self.dependency_predictions,
                self.dependency_type_predictions,
            ],
        )
        documents = [documents[example_id] for example_id in example_ids]

        base_phrase_based_metrics = conll18_ud_eval(
            *self._to_base_phrase_based_conll(documents, dependency_predictions, dependency_type_predictions)
        )
        morpheme_based_metrics = conll18_ud_eval(
            *self._to_morpheme_based_conll(documents, dependency_predictions, dependency_type_predictions)
        )
        metrics = {"base_phrase_" + key: value for key, value in base_phrase_based_metrics.items()}
        metrics.update({"morpheme_" + key: value for key, value in morpheme_based_metrics.items()})
        return metrics

    @staticmethod
    def unique(x: torch.Tensor, dim: int = None):
        unique, inverse = torch.unique(x, sorted=True, return_inverse=True, dim=dim)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        return inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)

    def _to_base_phrase_based_conll(
        self,
        documents: list[Document],
        dependency_predictions: torch.Tensor,
        dependency_type_predictions: torch.Tensor,
    ) -> tuple[list[str], list[str]]:
        gold_lines, system_lines = [], []
        for document, dependency_prediction, dependency_type_prediction in zip(
            documents, dependency_predictions, dependency_type_predictions
        ):
            sequence_len = len(dependency_prediction)
            for sentence in extract_target_sentences(document):
                morpheme_global_index2base_phrase_index = {
                    morpheme.global_index: base_phrase.index + 1
                    for base_phrase in sentence.base_phrases
                    for morpheme in base_phrase.morphemes
                }
                morpheme_global_index2base_phrase_index[sequence_len - 1] = 0
                dependency_manager = DependencyManager()

                for base_phrase in sentence.base_phrases:
                    gold_head = base_phrase.parent_index + 1
                    gold_deprel = base_phrase.dep_type
                    gold_lines.append(self._to_conll_line(base_phrase, gold_head, gold_deprel))

                    # goldの基本句主辞
                    system_head, system_deprel = self.get_system_predictions(
                        base_phrase,
                        dependency_prediction[base_phrase.head.global_index],
                        dependency_type_prediction[base_phrase.head.global_index],
                        morpheme_global_index2base_phrase_index,
                        dependency_manager,
                    )
                    system_lines.append(self._to_conll_line(base_phrase, system_head, system_deprel))
                    if system_head == 0:
                        dependency_manager.root = True

                gold_lines.append("\n")
                system_lines.append("\n")
        return gold_lines, system_lines

    def _to_morpheme_based_conll(
        self,
        documents: list[Document],
        dependency_predictions: torch.Tensor,
        dependency_type_predictions: torch.Tensor,
    ) -> tuple[list[str], list[str]]:
        gold_lines, system_lines = [], []
        for document, dependency_prediction, dependency_type_prediction in zip(
            documents, dependency_predictions, dependency_type_predictions
        ):
            sequence_len = len(dependency_prediction)
            for sentence in extract_target_sentences(document):
                morpheme_global_index2morpheme_index = {
                    morpheme.global_index: morpheme.index + 1 for morpheme in sentence.morphemes
                }
                morpheme_global_index2morpheme_index[sequence_len - 1] = 0
                dependency_manager = DependencyManager()

                for morpheme in sentence.morphemes:
                    gold_head = morpheme.parent.index + 1 if morpheme.parent else 0
                    if morpheme == morpheme.base_phrase.head:
                        gold_deprel = morpheme.base_phrase.dep_type
                    else:
                        gold_deprel = DepType.DEPENDENCY
                    gold_lines.append(self._to_conll_line(morpheme, gold_head, gold_deprel))

                    system_head, system_deprel = self.get_system_predictions(
                        morpheme,
                        dependency_prediction[morpheme.global_index],
                        dependency_type_prediction[morpheme.global_index],
                        morpheme_global_index2morpheme_index,
                        dependency_manager,
                    )

                    system_lines.append(self._to_conll_line(morpheme, system_head, system_deprel))
                    if system_head == 0:
                        dependency_manager.root = True

                gold_lines.append("\n")
                system_lines.append("\n")
        return gold_lines, system_lines

    @staticmethod
    def _to_conll_line(unit: Union[BasePhrase, Morpheme], head: int, deprel: DepType) -> str:
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
            map(
                str,
                [id_, form, lemma, upos, xpos, feats, head, deprel.value, deps, misc],
            )
        )

    @staticmethod
    def get_number_of_units(unit: Union[BasePhrase, Morpheme]) -> int:
        if isinstance(unit, BasePhrase):
            return len(unit.sentence.base_phrases)
        elif isinstance(unit, Morpheme):
            return len(unit.sentence.morphemes)
        else:
            raise TypeError

    def get_system_predictions(
        self,
        unit: Union[BasePhrase, Morpheme],
        topk_heads: torch.Tensor,  # (k, )
        topk_dependency_types: torch.Tensor,  # (k, )
        morpheme_global_index2unit_index: dict[int, int],
        dependency_manager: DependencyManager,
    ) -> tuple[int, DepType]:
        src = unit.index + 1
        for head, dependency_type in zip(topk_heads, topk_dependency_types):
            system_head = morpheme_global_index2unit_index[head]
            system_deprel = INDEX2DEPENDENCY_TYPE[dependency_type]
            dependency_manager.add_edge(src, system_head)
            if dependency_manager.has_cycle() or (system_head == 0 and dependency_manager.root):
                dependency_manager.remove_edge(src, system_head)
            else:
                break
        else:
            # 末尾の基本句/形態素まで見てROOTがない時
            if src == self.get_number_of_units(unit) and not dependency_manager.root:
                system_head, system_deprel = 0, DepType.DEPENDENCY
            else:
                system_head, system_deprel = self._resolve_dependency(unit, dependency_manager)
        return system_head, system_deprel

    def _resolve_dependency(
        self, unit: Union[BasePhrase, Morpheme], dependency_manager: DependencyManager
    ) -> tuple[int, DepType]:
        src = unit.index + 1
        num_units = self.get_number_of_units(unit)
        # 日本語の係り受けは基本的にleft-to-rightなので、まず右方向に係れるか調べる
        for dst in range(src + 1, num_units + 1):
            dependency_manager.add_edge(src, dst)
            if dependency_manager.has_cycle():
                dependency_manager.remove_edge(src, dst)
            else:
                return dst, DepType.DEPENDENCY

        for dst in range(src - 1, 0, -1):
            dependency_manager.add_edge(src, dst)
            if dependency_manager.has_cycle():
                dependency_manager.remove_edge(src, dst)
            else:
                return dst, DepType.DEPENDENCY

        raise RuntimeError("couldn't resolve dependency")
