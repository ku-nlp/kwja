from typing import Dict

import torch

from kwja.metrics.base import BaseModuleMetric
from kwja.metrics.utils import unique


class Seq2SeqModuleMetric(BaseModuleMetric):
    STATE_NAMES = (
        "example_ids",
        "loss",
    )

    def __init__(self, max_seq_length: int) -> None:
        super().__init__(max_seq_length)
        self.example_ids: torch.Tensor
        self.loss: torch.Tensor

    def _pad(self, kwargs: Dict[str, torch.Tensor]) -> None:
        pass

    def compute(self) -> Dict[str, float]:
        if isinstance(self.example_ids, torch.Tensor) is False:
            self.example_ids = torch.cat(self.example_ids, dim=0)  # type: ignore
        sorted_indices = unique(self.example_ids)
        for state_name in self.STATE_NAMES:
            state = getattr(self, state_name)
            if isinstance(state, torch.Tensor) is False:
                if state_name != "loss":
                    state = torch.cat(state, dim=0)
                else:
                    state = torch.stack(state, dim=0)
                    setattr(self, state_name, state[sorted_indices])
            if state_name != "loss":
                setattr(self, state_name, state[sorted_indices])

        return {"seq2seq_loss": self.loss.mean().item()}
