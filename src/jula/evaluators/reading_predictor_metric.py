from typing import Union

import torch
from sklearn.metrics import accuracy_score
from torchmetrics import Metric

from jula.utils.constants import IGNORE_INDEX
from jula.utils.reading import UNK_ID


class ReadingPredictorMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("predictions", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        self.predictions.append(predictions)
        self.labels.append(labels)

    def compute(self) -> dict[str, Union[torch.Tensor, float]]:
        predictions = self.predictions.view(-1)
        labels = self.labels.view(-1)
        ignore_indexes = labels == IGNORE_INDEX
        predictions = predictions[~ignore_indexes]
        labels = labels[~ignore_indexes]
        acc = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        unk_indexes = predictions == UNK_ID
        predictions = predictions[~unk_indexes]
        labels = labels[~unk_indexes]
        acc_unk = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        return {"reading_predictor_accuracy": acc - acc_unk}
