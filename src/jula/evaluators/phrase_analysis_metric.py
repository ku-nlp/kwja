from itertools import chain
from typing import Union

import numpy as np
import torch
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2
from torchmetrics import Metric

from jula.utils.utils import BASE_PHRASE_FEATURES, IGNORE_INDEX


class PhraseAnalysisMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("example_ids", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")

    def update(
        self, example_ids: torch.Tensor, preds: torch.Tensor, labels: torch.Tensor
    ):
        self.example_ids.append(example_ids)
        self.preds.append(preds)
        self.labels.append(labels)

    def compute(self) -> dict[str, Union[torch.Tensor, float]]:
        sorted_indices = self.unique(self.example_ids)
        # (num_base_phrase_features, batch_size, max_seq_len)
        preds, labels = map(
            lambda x: x[sorted_indices].permute(2, 0, 1), [self.preds, self.labels]
        )

        metrics, num_base_phrase_features = {}, 0
        for i, (pred, label) in enumerate(zip(preds, labels)):
            # pred/label: (batch_size, seq_len)
            aligned_pred, aligned_label = self.align_predictions(
                pred=pred.detach().cpu().numpy(),
                label=label.detach().cpu().numpy(),
            )
            if (
                sum(bio_tag == "B" for bio_tag in chain.from_iterable(aligned_label))
                > 0
            ):
                metrics[f"{BASE_PHRASE_FEATURES[i]}_f1"] = f1_score(
                    y_true=aligned_label,
                    y_pred=aligned_pred,
                    mode="strict",
                    scheme=IOB2,
                    zero_division=0,
                )
                num_base_phrase_features += 1
        else:
            sum_f1 = sum(value for key, value in metrics.items() if key.endswith("_f1"))
            metrics["phrase_analysis_f1"] = sum_f1 / num_base_phrase_features
        return metrics

    @staticmethod
    def unique(x, dim=None):
        unique, inverse = torch.unique(x, sorted=True, return_inverse=True, dim=dim)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        return inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)

    @staticmethod
    def align_predictions(
        pred: np.ndarray, label: np.ndarray
    ) -> tuple[list[list[str]], list[list[str]]]:
        batch_size, seq_len = pred.shape
        aligned_pred: list[list[str]] = [[] for _ in range(batch_size)]
        aligned_label: list[list[str]] = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(seq_len):
                if label[i, j] != float(IGNORE_INDEX):
                    aligned_pred[i].append("B" if pred[i][j] == 1.0 else "I")
                    aligned_label[i].append("B" if label[i][j] == 1.0 else "I")
        return aligned_pred, aligned_label
