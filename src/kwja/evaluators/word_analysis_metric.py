import torch
from torchmetrics import Metric
from torchmetrics.functional import f1_score

from kwja.utils.constants import IGNORE_INDEX


class WordAnalysisMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("pos_preds", default=[], dist_reduce_fx="cat")
        self.add_state("pos_labels", default=[], dist_reduce_fx="cat")
        self.add_state("subpos_preds", default=[], dist_reduce_fx="cat")
        self.add_state("subpos_labels", default=[], dist_reduce_fx="cat")
        self.add_state("conjtype_preds", default=[], dist_reduce_fx="cat")
        self.add_state("conjtype_labels", default=[], dist_reduce_fx="cat")
        self.add_state("conjform_preds", default=[], dist_reduce_fx="cat")
        self.add_state("conjform_labels", default=[], dist_reduce_fx="cat")

    def update(
        self,
        pos_preds: torch.Tensor,
        pos_labels: torch.Tensor,
        subpos_preds: torch.Tensor,
        subpos_labels: torch.Tensor,
        conjtype_preds: torch.Tensor,
        conjtype_labels: torch.Tensor,
        conjform_preds: torch.Tensor,
        conjform_labels: torch.Tensor,
    ) -> None:
        self.pos_preds.append(pos_preds)
        self.pos_labels.append(pos_labels)
        self.subpos_preds.append(subpos_preds)
        self.subpos_labels.append(subpos_labels)
        self.conjtype_preds.append(conjtype_preds)
        self.conjtype_labels.append(conjtype_labels)
        self.conjform_preds.append(conjform_preds)
        self.conjform_labels.append(conjform_labels)

    @staticmethod
    def _convert(preds: torch.Tensor, labels: torch.Tensor):
        ignore_index_pos: torch.BoolTensor = labels != IGNORE_INDEX
        converted_preds = torch.masked_select(preds, ignore_index_pos)
        converted_labels = torch.masked_select(labels, ignore_index_pos)
        return converted_preds, converted_labels

    def compute(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        pos_preds, pos_labels = self._convert(preds=self.pos_preds, labels=self.pos_labels)
        metrics["pos_f1"] = f1_score(
            preds=pos_preds,
            target=pos_labels,
        ).item()
        subpos_preds, subpos_labels = self._convert(preds=self.subpos_preds, labels=self.subpos_labels)
        metrics["subpos_f1"] = f1_score(
            preds=subpos_preds,
            target=subpos_labels,
        ).item()
        conjtype_preds, conjtype_labels = self._convert(preds=self.conjtype_preds, labels=self.conjtype_labels)
        metrics["conjtype_f1"] = f1_score(
            preds=conjtype_preds,
            target=conjtype_labels,
        ).item()
        conjform_preds, conjform_labels = self._convert(preds=self.conjform_preds, labels=self.conjform_labels)
        metrics["conjform_f1"] = f1_score(
            preds=conjform_preds,
            target=conjform_labels,
        ).item()
        f1s: list[float] = [
            metrics["pos_f1"],
            metrics["subpos_f1"],
            metrics["conjtype_f1"],
            metrics["conjform_f1"],
        ]
        metrics["word_analysis_f1"] = sum(f1s) / len(f1s)

        return metrics
