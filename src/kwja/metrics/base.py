from typing import Any, Dict

import torch
from torchmetrics import Metric


class BaseModuleMetric(Metric):
    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        for state_name in self.STATE_NAMES:
            self.add_state(state_name, default=[], dist_reduce_fx="cat")

    def update(self, kwargs: Dict[str, torch.Tensor]) -> None:
        for state_name in self.STATE_NAMES:
            state = getattr(self, state_name)
            state.append(kwargs[state_name])

    def set_properties(self, kwargs: Dict[str, Any]) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
