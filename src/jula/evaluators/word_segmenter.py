from typing import Tuple, Union

import torch
from seqeval.metrics import accuracy_score, f1_score
from seqeval.scheme import IOB2
from torchmetrics import Metric

from jula.utils.utils import IGNORE_INDEX, INDEX2SEG_TYPE


class WordSegmenterMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("seg_preds", default=[], dist_reduce_fx="cat")
        self.add_state("seg_labels", default=[], dist_reduce_fx="cat")

    @staticmethod
    def filter_predictions(
        preds: list[str], labels: list[str]
    ) -> Tuple[list[str], list[str]]:
        filtered_preds = []
        filtered_labels = []
        for pred, label in zip(preds, labels):
            if label == "O":
                continue
            filtered_preds.append(pred)
            filtered_labels.append(label)
        return filtered_preds, filtered_labels

    @staticmethod
    def convert_num2label(
        preds: list[list[int]], labels: list[list[int]]
    ) -> Tuple[list[list[str]], list[list[str]]]:
        converted_preds: list[list[str]] = []
        converted_labels: list[list[str]] = []
        for pred, label in zip(preds, labels):
            converted_pred: list[str] = []
            converted_label: list[str] = []
            for pred_num, label_num in zip(pred, label):
                if label_num == IGNORE_INDEX:
                    continue
                converted_pred.append(INDEX2SEG_TYPE[pred_num])
                converted_label.append(INDEX2SEG_TYPE[label_num])
            converted_preds.append(converted_pred)
            converted_labels.append(converted_label)
        return converted_preds, converted_labels

    def update(self, seg_preds: torch.Tensor, seg_labels: torch.Tensor) -> None:
        self.seg_preds.append(seg_preds)
        self.seg_labels.append(seg_labels)

    def compute(self):
        metrics: dict[str, Union[torch.Tensor, float]] = dict()
        seg_preds, seg_labels = self.convert_num2label(
            preds=self.seg_preds.cpu().tolist(),
            labels=self.seg_labels.cpu().tolist(),
        )  # (b, seq_len), (b, seq_len)
        metrics["word_segmenter_acc"] = accuracy_score(
            y_true=seg_labels, y_pred=seg_preds
        )
        metrics["word_segmenter_f1"] = f1_score(
            y_true=seg_labels,
            y_pred=seg_preds,
            mode="strict",
            scheme=IOB2,
            zero_division=0,
        )
        return metrics
