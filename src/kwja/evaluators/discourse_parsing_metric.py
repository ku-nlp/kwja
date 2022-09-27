from typing import Union

import torch
from torchmetrics import Metric
from torchmetrics.functional import accuracy

from kwja.utils.constants import DISCOURSE_RELATIONS, IGNORE_INDEX


class DiscourseParsingMetric(Metric):
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        # Metric state variables can either be torch.Tensor or an empty list which can be used to store torch.Tensors`.
        # i.e. Expected metric state to be either a Tensor or a list of Tensor
        self.add_state("discourse_parsing_predictions", default=list(), dist_reduce_fx="cat")  # list[torch.Tensor]
        self.add_state("discourse_parsing_labels", default=list(), dist_reduce_fx="cat")  # list[torch.Tensor]

    def update(
        self,
        discourse_parsing_predictions: torch.Tensor,  # (b, seq, seq)
        discourse_parsing_labels: torch.Tensor,  # (b, seq, seq)
    ) -> None:
        self.discourse_parsing_predictions.append(discourse_parsing_predictions)
        self.discourse_parsing_labels.append(discourse_parsing_labels)

    def compute(self) -> dict[str, Union[torch.Tensor, float]]:
        predictions = self.discourse_parsing_predictions.view(-1)
        labels = self.discourse_parsing_labels.view(-1)

        ignored_indexes = labels == IGNORE_INDEX
        predictions = predictions[~ignored_indexes]
        labels = labels[~ignored_indexes]
        if labels.numel() == 0:
            acc = 0.0
        else:
            acc = accuracy(predictions, labels).item()

        no_relation_index = DISCOURSE_RELATIONS.index("談話関係なし")
        # Precision
        ignored_indexes = predictions == no_relation_index
        if predictions[~ignored_indexes].numel() == 0:
            prec = 0.0
        else:
            prec = accuracy(predictions[~ignored_indexes], labels[~ignored_indexes]).item()
        # Recall
        ignored_indexes = labels == no_relation_index
        if labels[~ignored_indexes].numel() == 0:
            rec = 0.0
        else:
            rec = accuracy(predictions[~ignored_indexes], labels[~ignored_indexes]).item()
        # F1
        if prec + rec == 0:
            f1 = 0.0
        else:
            f1 = 2 * prec * rec / (prec + rec)
        return {
            "discourse_parsing_acc": acc,
            "discourse_parsing_precision": prec,
            "discourse_parsing_recall": rec,
            "discourse_parsing_f1": f1,
        }
