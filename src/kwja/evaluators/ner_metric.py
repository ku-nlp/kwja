from typing import Union

import numpy as np
import torch
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2
from torchmetrics import Metric

from kwja.utils.constants import IGNORE_INDEX, NE_TAGS


class NERMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("example_ids", default=[], dist_reduce_fx="cat")
        self.add_state("ne_tag_predictions", default=[], dist_reduce_fx="cat")
        self.add_state("ne_tags", default=[], dist_reduce_fx="cat")

    def update(
        self,
        example_ids: torch.Tensor,
        ne_tag_predictions: torch.Tensor,
        ne_tags: torch.Tensor,
    ):
        self.example_ids.append(example_ids)
        self.ne_tag_predictions.append(ne_tag_predictions)
        self.ne_tags.append(ne_tags)

    def compute(self) -> dict[str, Union[torch.Tensor, float]]:
        sorted_indices = self._unique(self.example_ids)
        ne_tag_predictions, ne_tags = map(
            lambda x: x[sorted_indices],
            [
                self.ne_tag_predictions,
                self.ne_tags,
            ],
        )

        metrics = self._compute_ner_metrics(ne_tag_predictions, ne_tags)
        return metrics

    @staticmethod
    def _unique(x: torch.Tensor, dim: int = None):
        unique, inverse = torch.unique(x, sorted=True, return_inverse=True, dim=dim)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        return inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)

    @staticmethod
    def _align_predictions(prediction: np.ndarray, label: np.ndarray) -> tuple[list[list[str]], list[list[str]]]:
        batch_size, sequence_len = label.shape
        aligned_prediction: list[list[str]] = [[] for _ in range(batch_size)]
        aligned_label: list[list[str]] = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(sequence_len):
                # 評価対象外の label / prediction は含めない
                if label[i, j] != IGNORE_INDEX:
                    aligned_prediction[i].append(NE_TAGS[prediction[i][j]])
                    aligned_label[i].append(NE_TAGS[label[i][j]])
        return aligned_prediction, aligned_label

    def _compute_ner_metrics(
        self,
        ne_tag_predictions: torch.Tensor,
        ne_tags: torch.Tensor,
    ) -> dict[str, float]:
        ner_metric = {}
        aligned_predictions, aligned_labels = self._align_predictions(
            prediction=ne_tag_predictions.detach().cpu().numpy(),
            label=ne_tags.detach().cpu().numpy(),
        )
        ner_metric["ner_f1"] = f1_score(
            y_true=aligned_labels,
            y_pred=aligned_predictions,
            mode="strict",
            zero_division=0,
            scheme=IOB2,
        )
        return ner_metric
