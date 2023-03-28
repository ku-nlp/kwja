from abc import ABC
from typing import Any, Dict, Tuple

import torch
from torchmetrics import Metric


class BaseModuleMetric(Metric, ABC):
    full_state_update = False
    STATE_NAMES: Tuple[str, ...]

    def __init__(self) -> None:
        super().__init__()
        for state_name in self.STATE_NAMES:
            self.add_state(state_name, default=[], dist_reduce_fx="cat")

    def update(self, kwargs: Dict[str, torch.Tensor]) -> None:
        for state_name in self.STATE_NAMES:
            state = getattr(self, state_name)
            value = kwargs[state_name]
            # https://github.com/pytorch/pytorch/issues/90245
            if isinstance(value, torch.BoolTensor):
                value = value.long()
            if len(value.size()) == 0:
                value = value.unsqueeze(dim=0)
            if isinstance(state, torch.Tensor):
                new_value = torch.cat([state, value], dim=0)
            else:
                assert len(state) == 0, f"state {state_name} is not empty"
                new_value = value
            setattr(self, state_name, new_value)

    def set_properties(self, kwargs: Dict[str, Any]) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
