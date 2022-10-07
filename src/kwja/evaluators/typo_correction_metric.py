from typing import Dict, Union

import torch
from torchmetrics.functional import accuracy


class TypoCorrectionMetric:
    def __init__(self) -> None:
        pass

    @staticmethod
    def compute_metrics(
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, Union[torch.Tensor, float]]:

        metrics: Dict[str, Union[torch.Tensor, float]] = dict(loss=outputs["loss"])
        for key in outputs.keys():
            if "_loss" in key:
                metrics[key] = outputs[key].detach()

        metrics["kdr_acc"] = accuracy(
            preds=torch.argmax(outputs["kdr_logits"], dim=-1),
            target=batch["kdr_labels"],
            ignore_index=0,
        )
        metrics["ins_acc"] = accuracy(
            preds=torch.argmax(outputs["ins_logits"], dim=-1),
            target=batch["ins_labels"],
            ignore_index=0,
        )

        return metrics
