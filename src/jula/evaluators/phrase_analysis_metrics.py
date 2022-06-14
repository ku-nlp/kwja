from typing import Union

import torch
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2

from jula.utils.utils import BASE_PHRASE_FEATURES


class PhraseAnalysisMetrics:
    def __init__(self) -> None:
        pass

    @staticmethod
    def align(
        preds: list[list[float]], labels: list[list[float]]
    ) -> tuple[list[list[str]], list[list[str]]]:
        aligned_preds: list[list[str]] = []
        aligned_labels: list[list[str]] = []
        for pred, label in zip(preds, labels):
            aligned_pred: list[str] = []
            aligned_label: list[str] = []
            for pred_num, label_num in zip(pred, label):
                if label_num == -100.:
                    continue
                aligned_pred.append('B' if pred_num == 1. else 'I')
                aligned_label.append('B' if label_num == 1. else 'I')
            aligned_preds.append(aligned_pred)
            aligned_labels.append(aligned_label)
        return aligned_preds, aligned_labels

    def compute_metrics(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, Union[torch.Tensor, float]]:
        metrics = dict(loss=output["phrase_analysis_loss"].item())
        # (num_base_phrase_features, batch_size, max_seq_len)
        preds = torch.where(output["phrase_analysis_logits"] >= 0.5, 1., 0.).permute(2, 0, 1)
        labels = batch["base_phrase_features"].permute(2, 0, 1)
        for index, (pred, label) in enumerate(zip(preds, labels)):
            # pred/label: (batch_size, seq_len)
            pred, label = self.align(
                preds=pred.cpu().tolist(),
                labels=label.cpu().tolist(),
            )
            metrics[f"{BASE_PHRASE_FEATURES[index]}_f1"] = f1_score(
                y_true=label,
                y_pred=pred,
                mode="strict",
                scheme=IOB2,
                zero_division=0,
            )
        else:
            sum_f1 = sum(value for key, value in metrics.items() if key.endswith("_f1"))
            metrics["phrase_analysis_f1"] = sum_f1 / len(BASE_PHRASE_FEATURES)
        return metrics
