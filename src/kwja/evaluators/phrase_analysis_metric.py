from itertools import chain
from typing import Literal

import numpy as np
import torch
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2
from torchmetrics import Metric

from kwja.utils.constants import BASE_PHRASE_FEATURES, IGNORE_INDEX, SUB_WORD_FEATURES, WORD_FEATURES


class PhraseAnalysisMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("example_ids", default=[], dist_reduce_fx="cat")
        self.add_state("word_feature_predictions", default=[], dist_reduce_fx="cat")
        self.add_state("word_features", default=[], dist_reduce_fx="cat")
        self.add_state("base_phrase_feature_predictions", default=[], dist_reduce_fx="cat")
        self.add_state("base_phrase_features", default=[], dist_reduce_fx="cat")

    def update(
        self,
        example_ids: torch.Tensor,
        word_feature_predictions: torch.Tensor,
        word_features: torch.Tensor,
        base_phrase_feature_predictions: torch.Tensor,
        base_phrase_features: torch.Tensor,
    ):
        self.example_ids.append(example_ids)
        self.word_feature_predictions.append(word_feature_predictions)
        self.word_features.append(word_features)
        self.base_phrase_feature_predictions.append(base_phrase_feature_predictions)
        self.base_phrase_features.append(base_phrase_features)

    def compute(self) -> dict[str, float]:
        sorted_indices = self.unique(self.example_ids)
        # (num_base_phrase_features, b, seq)
        (word_feature_predictions, word_features, base_phrase_feature_predictions, base_phrase_features,) = map(
            lambda x: x[sorted_indices].permute(2, 0, 1),
            [
                self.word_feature_predictions,
                self.word_features,
                self.base_phrase_feature_predictions,
                self.base_phrase_features,
            ],
        )

        metrics = self.compute_word_feature_metrics(word_feature_predictions, word_features)
        metrics.update(self.compute_base_phrase_feature_metrics(base_phrase_feature_predictions, base_phrase_features))
        return metrics

    @staticmethod
    def unique(x: torch.Tensor, dim: int = None):
        unique, inverse = torch.unique(x, sorted=True, return_inverse=True, dim=dim)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        return inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)

    @staticmethod
    def align_predictions(
        prediction: np.ndarray, label: np.ndarray, io_tag: Literal["I", "O"]
    ) -> tuple[list[list[str]], list[list[str]]]:
        b, seq_len = label.shape
        aligned_prediction: list[list[str]] = [[] for _ in range(b)]
        aligned_label: list[list[str]] = [[] for _ in range(b)]
        for i in range(b):
            for j in range(seq_len):
                # 評価対象外の label / prediction は含めない
                if label[i, j] != IGNORE_INDEX:
                    aligned_prediction[i].append("B" if prediction[i][j] == 1 else io_tag)
                    aligned_label[i].append("B" if label[i][j] == 1 else io_tag)
        return aligned_prediction, aligned_label

    def compute_word_feature_metrics(
        self,
        word_feature_predictions: torch.Tensor,
        word_features: torch.Tensor,
    ) -> dict[str, float]:
        word_feature_metrics = {}
        aligned_predictions, aligned_labels = [], []
        for i, (word_feature_prediction, word_feature_label) in enumerate(zip(word_feature_predictions, word_features)):
            if i < len(WORD_FEATURES) - len(SUB_WORD_FEATURES):
                io_tag: Literal["I", "O"] = "I"
            else:
                io_tag = "O"

            # prediction / label: (b, seq)
            aligned_prediction, aligned_label = self.align_predictions(
                prediction=word_feature_prediction.detach().cpu().numpy(),
                label=word_feature_label.detach().cpu().numpy(),
                io_tag=io_tag,
            )
            aligned_predictions += aligned_prediction
            aligned_labels += aligned_label
            word_feature_metrics[f"{WORD_FEATURES[i]}_f1"] = f1_score(
                y_true=aligned_label,
                y_pred=aligned_prediction,
                mode="strict",
                zero_division=0,
                scheme=IOB2,
            )
        else:
            sum_f1 = sum(value for key, value in word_feature_metrics.items())
            word_feature_metrics["macro_word_feature_f1"] = sum_f1 / len(word_feature_metrics)
            word_feature_metrics["micro_word_feature_f1"] = f1_score(
                y_true=aligned_labels,
                y_pred=aligned_predictions,
                mode="strict",
                zero_division=0,
                scheme=IOB2,
            )
        return word_feature_metrics

    def compute_base_phrase_feature_metrics(
        self,
        base_phrase_feature_predictions: torch.Tensor,
        base_phrase_features: torch.Tensor,
    ) -> dict[str, float]:
        base_phrase_feature_metrics = {}
        aligned_predictions, aligned_labels = [], []
        for i, (base_phrase_feature_prediction, base_phrase_feature_label) in enumerate(
            zip(base_phrase_feature_predictions, base_phrase_features)
        ):
            # prediction / label: (b, seq)
            aligned_prediction, aligned_label = self.align_predictions(
                prediction=base_phrase_feature_prediction.detach().cpu().numpy(),
                label=base_phrase_feature_label.detach().cpu().numpy(),
                io_tag="O",
            )
            t = sum(bio_tag == "B" for bio_tag in chain.from_iterable(aligned_label))
            # 正解ラベルがない基本句素性は評価対象外
            if t > 0:
                aligned_predictions += aligned_prediction
                aligned_labels += aligned_label
                base_phrase_feature_metrics[f"{BASE_PHRASE_FEATURES[i]}_f1"] = f1_score(
                    y_true=aligned_label,
                    y_pred=aligned_prediction,
                    mode="strict",
                    zero_division=0,
                    scheme=IOB2,
                )
        else:
            sum_f1 = sum(value for key, value in base_phrase_feature_metrics.items())
            base_phrase_feature_metrics["macro_base_phrase_feature_f1"] = sum_f1 / len(base_phrase_feature_metrics)
            base_phrase_feature_metrics["micro_base_phrase_feature_f1"] = f1_score(
                y_true=aligned_labels,
                y_pred=aligned_predictions,
                mode="strict",
                zero_division=0,
                scheme=IOB2,
            )
        return base_phrase_feature_metrics
