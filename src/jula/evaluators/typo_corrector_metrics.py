from typing import Union

import torch
from torchmetrics import Accuracy


class TypoCorrectorMetrics:
    def __init__(self) -> None:
        self.char_acc = Accuracy(mdmc_average="global", ignore_index=0)

    def compute_metrics(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, Union[torch.Tensor, float]]:

        metrics = dict(loss=outputs["loss"])
        for key in outputs.keys():
            if "_loss" in key:
                metrics[key] = outputs[key].detach()

        metrics["kdr_acc"] = self.char_acc(
            torch.argmax(outputs["kdr_logits"], dim=-1).cpu(), batch["kdr_labels"].cpu()
        )
        metrics["ins_acc"] = self.char_acc(
            torch.argmax(outputs["ins_logits"], dim=-1).cpu(), batch["ins_labels"].cpu()
        )

        return metrics
