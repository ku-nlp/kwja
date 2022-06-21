from typing import Tuple, Union

import torch
from seqeval.metrics import accuracy_score, f1_score
from seqeval.scheme import IOB2

from jula.utils.utils import INDEX2SEG_LABEL, SEG_LABEL2INDEX


class WordSegmenterMetric:
    def __init__(self) -> None:
        pass

    @staticmethod
    def convert_ids_to_labels(ids: list[int]) -> list[str]:
        return [
            INDEX2SEG_LABEL[id_] if id_ != SEG_LABEL2INDEX["PAD"] else "O"
            for id_ in ids
        ]

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
                if label_num == SEG_LABEL2INDEX["PAD"]:
                    continue
                converted_pred.append(
                    INDEX2SEG_LABEL[pred_num] if pred_num != 0 else "O"
                )
                converted_label.append(INDEX2SEG_LABEL[label_num])
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

        seg_preds = [
            self.convert_ids_to_labels(ids)
            for ids in torch.argmax(outputs["logits"], dim=-1).cpu().tolist()
        ]
        seg_labels = [
            self.convert_ids_to_labels(ids)
            for ids in batch["seg_labels"].cpu().tolist()
        ]
        filtered_seg_preds = []
        filtered_seg_labels = []
        for seg_preds_item, seg_labels_item in zip(seg_preds, seg_labels):
            filtered_seg_preds_item, filtered_seg_labels_item = self.filter_predictions(
                seg_preds_item, seg_labels_item
            )
            filtered_seg_preds.append(filtered_seg_preds_item)
            filtered_seg_labels.append(filtered_seg_labels_item)
        metrics["acc"] = accuracy_score(
            y_true=filtered_seg_labels, y_pred=filtered_seg_preds
        )
        metrics["f1"] = f1_score(
            y_true=filtered_seg_labels,
            y_pred=filtered_seg_preds,
            mode="strict",
            scheme=IOB2,
            zero_division=0,
        )
        return metrics
