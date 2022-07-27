from typing import Union

import torch
from torchmetrics import Metric
from torchmetrics.functional import f1_score

from jula.utils.constants import DISCOURSE_RELATIONS, IGNORE_INDEX


class DiscourseParsingMetric(Metric):
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        # Metric state variables can either be torch.Tensor or an empty list which can be used to store torch.Tensors`.
        # i.e. Expected metric state to be either a Tensor or a list of Tensor
        self.add_state("predictions", default=list(), dist_reduce_fx="cat")  # list[torch.Tensor]
        self.add_state("labels", default=list(), dist_reduce_fx="cat")  # list[torch.Tensor]

    def update(
        self,
        discourse_parsing_predictions: torch.Tensor,  # (b, seq, seq)
        discourse_parsing_labels: torch.Tensor,  # (b, seq, seq)
    ) -> None:
        self.predictions.append(discourse_parsing_predictions)
        self.labels.append(discourse_parsing_labels)

    @staticmethod
    def _filter_ignore_index(t: torch.Tensor, labels: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
        return torch.masked_select(t, labels != ignore_index)

    def compute(self) -> dict[str, Union[torch.Tensor, float]]:
        predictions = self.predictions.view(-1)
        labels = self.labels.view(-1)

        predictions = self._filter_ignore_index(predictions, labels)
        labels = self._filter_ignore_index(labels, labels)
        if labels.numel() == 0:
            discourse_parsing_f1 = 0.0
        else:
            discourse_parsing_f1 = f1_score(predictions, labels)

        no_relation_index = DISCOURSE_RELATIONS.index("談話関係なし")
        predictions = self._filter_ignore_index(predictions, labels, ignore_index=no_relation_index)
        labels = self._filter_ignore_index(labels, labels, ignore_index=no_relation_index)
        if labels.numel() == 0:
            discourse_parsing_f1_no_relation_ignored = 0.0
        else:
            discourse_parsing_f1_no_relation_ignored = f1_score(predictions, labels)

        return {
            "discourse_parsing_f1": discourse_parsing_f1,
            "discourse_parsing_f1_no_relation_ignored": discourse_parsing_f1_no_relation_ignored,
        }
