from typing import Dict

import torch

from kwja.metrics.base import BaseModuleMetric
from kwja.metrics.utils import unique


class Seq2SeqModuleMetric(BaseModuleMetric):
    STATE_NAMES = (
        "example_ids",
        "loss",
    )

    def __init__(self):
        super().__init__()

        self.example_ids: torch.Tensor
        self.loss: torch.Tensor

    def compute(self) -> Dict[str, float]:
        sorted_indices = unique(self.example_ids)
        for state_name in self.STATE_NAMES:
            state = getattr(self, state_name)
            if state_name != "loss":
                setattr(self, state_name, state[sorted_indices])

        metrics: Dict[str, float] = {
            "seq2seq_loss": self.loss.mean().item(),
        }
        return metrics
