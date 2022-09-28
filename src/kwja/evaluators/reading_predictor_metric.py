from typing import Union

import torch
from torchmetrics import Metric

from kwja.utils.constants import IGNORE_INDEX
from kwja.utils.reading import UNK_ID


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
        predictions = predictions[~ignore_indexes].cpu().numpy()
        labels = labels[~ignore_indexes].cpu().numpy()
        num_correct = sum(p == l and p != UNK_ID for p, l in zip(predictions, labels))
        return {"reading_prediction_accuracy": num_correct / len(predictions)}
