from typing import Tuple, Union

import numpy as np
import torch
from seqeval.metrics import accuracy_score, f1_score
from seqeval.scheme import IOB2
from torchmetrics import Metric

from kwja.utils.constants import IGNORE_INDEX, INDEX2SEG_TYPE


class WordSegmentationMetric(Metric):
    full_state_update: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("seg_preds", default=[], dist_reduce_fx="cat")
        self.add_state("seg_types", default=[], dist_reduce_fx="cat")

    @staticmethod
    def filter_predictions(preds: list[str], labels: list[str]) -> Tuple[list[str], list[str]]:
        filtered_preds = []
        filtered_labels = []
        for pred, label in zip(preds, labels):
            if label == "O":
                continue
            filtered_preds.append(pred)
            filtered_labels.append(label)
        return filtered_preds, filtered_labels

    @staticmethod
    def _align_predictions(prediction: np.ndarray, label: np.ndarray) -> tuple[list[list[str]], list[list[str]]]:
        batch_size, sequence_len = label.shape
        aligned_prediction: list[list[str]] = [[] for _ in range(batch_size)]
        aligned_label: list[list[str]] = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(sequence_len):
                # 評価対象外の label / prediction は含めない
                if label[i, j] != IGNORE_INDEX:
                    aligned_prediction[i].append(INDEX2SEG_TYPE[prediction[i][j]])
                    aligned_label[i].append(INDEX2SEG_TYPE[label[i][j]])
        return aligned_prediction, aligned_label

    def update(self, seg_preds: torch.Tensor, seg_types: torch.Tensor) -> None:
        self.seg_preds.append(seg_preds)
        self.seg_types.append(seg_types)

    def compute(self):
        metrics: dict[str, Union[torch.Tensor, float]] = dict()
        aligned_predictions, aligned_labels = self._align_predictions(
            prediction=self.seg_preds.detach().cpu().numpy(),
            label=self.seg_types.detach().cpu().numpy(),
        )
        metrics["word_segmenter_acc"] = accuracy_score(y_true=aligned_labels, y_pred=aligned_predictions)
        metrics["word_segmenter_f1"] = f1_score(
            y_true=aligned_labels,
            y_pred=aligned_predictions,
            mode="strict",
            scheme=IOB2,
            zero_division=0,
        )
        return metrics
