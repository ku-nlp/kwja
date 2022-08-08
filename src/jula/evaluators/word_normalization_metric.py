from typing import Union

import torch
from sklearn.metrics import accuracy_score, f1_score
from torchmetrics import Metric

from jula.utils.constants import IGNORE_INDEX, INDEX2CHARNORM_TYPE


class WordNormalizationMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("charnorm_preds", default=[], dist_reduce_fx="cat")
        self.add_state("charnorm_labels", default=[], dist_reduce_fx="cat")

    def update(self, charnorm_preds: torch.Tensor, charnorm_labels: torch.Tensor) -> None:
        ignore_indexes = charnorm_labels == IGNORE_INDEX
        self.charnorm_preds.append(charnorm_preds[~ignore_indexes])
        self.charnorm_labels.append(charnorm_labels[~ignore_indexes])

    def compute(self):
        metrics: dict[str, Union[torch.Tensor, float]] = dict()
        preds = self.charnorm_preds.cpu()
        labels = self.charnorm_labels.cpu()
        metrics["word_normalization_acc"] = accuracy_score(labels.tolist(), preds.tolist())
        metrics["word_normalization_f1"] = f1_score(labels.tolist(), preds.tolist(), average="micro")
        exist_charnorm_type = [
            charnorm_type
            for index, charnorm_type in INDEX2CHARNORM_TYPE.items()
            if index in set(labels.tolist() + preds.tolist())
        ]
        word_normalization_f1s = f1_score(labels.tolist(), preds.tolist(), average=None)
        assert len(word_normalization_f1s) == len(exist_charnorm_type)
        for charnorm_type, f1 in zip(exist_charnorm_type, word_normalization_f1s):
            metrics[f"word_normalization_f1_{charnorm_type}"] = f1
        for charnorm_type in INDEX2CHARNORM_TYPE.values():
            if f"word_normalization_f1_{charnorm_type}" not in metrics:
                metrics[f"word_normalization_f1_{charnorm_type}"] = 0.0
        return metrics
