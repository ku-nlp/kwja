from typing import Union

import torch
from sklearn.metrics import accuracy_score, f1_score
from torchmetrics import Metric

from jula.utils.constants import IGNORE_INDEX, INDEX2WORD_NORM_TYPE


class WordNormalizationMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("word_norm_preds", default=[], dist_reduce_fx="cat")
        self.add_state("word_norm_labels", default=[], dist_reduce_fx="cat")

    def update(self, word_norm_preds: torch.Tensor, word_norm_labels: torch.Tensor) -> None:
        ignore_indexes = word_norm_labels == IGNORE_INDEX
        self.word_norm_preds.append(word_norm_preds[~ignore_indexes])
        self.word_norm_labels.append(word_norm_labels[~ignore_indexes])

    def compute(self):
        metrics: dict[str, Union[torch.Tensor, float]] = dict()
        preds = self.word_norm_preds.cpu()
        labels = self.word_norm_labels.cpu()
        metrics["word_normalization_acc"] = accuracy_score(labels.tolist(), preds.tolist())
        metrics["word_normalization_f1"] = f1_score(labels.tolist(), preds.tolist(), average="micro")
        exist_word_norm_type = [
            word_norm_type
            for index, word_norm_type in INDEX2WORD_NORM_TYPE.items()
            if index in set(labels.tolist() + preds.tolist())
        ]
        word_normalization_f1s = f1_score(labels.tolist(), preds.tolist(), average=None)
        assert len(word_normalization_f1s) == len(exist_word_norm_type)
        for word_norm_type, f1 in zip(exist_word_norm_type, word_normalization_f1s):
            metrics[f"word_normalization_f1:{word_norm_type}"] = f1
        for word_norm_type in INDEX2WORD_NORM_TYPE.values():
            if f"word_normalization_f1_{word_norm_type}" not in metrics:
                metrics[f"word_normalization_f1:{word_norm_type}"] = 0.0
        return metrics
