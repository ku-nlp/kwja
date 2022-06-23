from typing import Tuple, Union

import torch
from seqeval.metrics import accuracy_score, f1_score
from seqeval.scheme import IOB2

from jula.utils.utils import IGNORE_INDEX, INDEX2SEG_TYPE


class WordSegmenterMetric:
    def __init__(self) -> None:
        pass

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

    def compute_metrics(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, Union[torch.Tensor, float]]:
        metrics: dict[str, Union[torch.Tensor, float]] = dict(loss=outputs["loss"])
        for key in outputs.keys():
            if "_loss" in key:
                metrics[key] = outputs[key].detach()

        seg_preds, seg_labels = self.convert_num2label(
            preds=torch.argmax(outputs["logits"], dim=-1).cpu().tolist(),
            labels=batch["seg_labels"].cpu().tolist(),
        )  # (b, seq_len), (b, seq_len)
        metrics["acc"] = accuracy_score(y_true=seg_labels, y_pred=seg_preds)
        metrics["f1"] = f1_score(
            y_true=seg_labels,
            y_pred=seg_preds,
            mode="strict",
            scheme=IOB2,
            zero_division=0,
        )
        return metrics
