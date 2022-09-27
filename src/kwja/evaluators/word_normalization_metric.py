from typing import Union

import torch
from sklearn.metrics import accuracy_score
from torchmetrics import Metric

from kwja.utils.constants import IGNORE_INDEX, INDEX2WORD_NORM_TYPE, WORD_NORM_TYPES


class WordNormalizationMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("norm_preds", default=[], dist_reduce_fx="cat")
        self.add_state("norm_types", default=[], dist_reduce_fx="cat")

    def update(self, norm_preds: torch.Tensor, norm_types: torch.Tensor) -> None:
        ignore_indexes = norm_types == IGNORE_INDEX
        self.norm_preds.append(norm_preds[~ignore_indexes])
        self.norm_types.append(norm_types[~ignore_indexes])

    def compute(self):
        metrics: dict[str, Union[torch.Tensor, float]] = dict()
        preds = self.norm_preds.cpu()
        labels = self.norm_types.cpu()

        # Accuracy
        metrics["word_normalization_accuracy"] = accuracy_score(labels.tolist(), preds.tolist())

        # Precision
        non_keep_indexes = preds != WORD_NORM_TYPES.index("K")
        if non_keep_indexes.sum() == 0:
            precision = 0.0
        else:
            precision = accuracy_score(preds[non_keep_indexes], labels[non_keep_indexes])
        metrics["word_normalization_precision"] = accuracy_score(labels.tolist(), preds.tolist())

        # Recall
        non_keep_indexes = labels != WORD_NORM_TYPES.index("K")
        if non_keep_indexes.sum() == 0:
            recall = 0.0
        else:
            recall = accuracy_score(preds[non_keep_indexes], labels[non_keep_indexes])
        metrics["word_normalization_recall"] = recall

        # F1
        if precision == 0.0 and recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        metrics["word_normalization_f1"] = f1

        # F1 score for each type
        for index, word_norm_type in INDEX2WORD_NORM_TYPE.items():
            # Precision
            indexes = preds == index
            if indexes.sum() == 0:
                precision = 0.0
            else:
                precision = accuracy_score(preds[indexes], labels[indexes])
            # Recall
            indexes = labels == index
            if indexes.sum() == 0:
                recall = 0.0
            else:
                recall = accuracy_score(preds[indexes], labels[indexes])
            # F1
            if precision == 0.0 and recall == 0.0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            metrics[f"word_normalization_f1:{word_norm_type}"] = f1
        return metrics
