from typing import Union

import torch
from rhoknp import Document
from torchmetrics import Metric


class DiscourseParsingMetric(Metric):
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        # Metric state variables can either be torch.Tensor or an empty list which can be used to store torch.Tensors`.
        # i.e. Expected metric state to be either a Tensor or a list of Tensor
        self.add_state("example_ids", default=list())  # list[torch.Tensor]
        self.add_state("discourse_parsing_predictions", default=list())  # list[torch.Tensor]  # [(rel, phrase)]

    def update(
        self,
        example_ids: torch.Tensor,  # (b)
        discourse_parsing_predictions: torch.Tensor,  # (b, seq, seq)
    ) -> None:
        self.example_ids.append(example_ids)
        self.discourse_parsing_predictions.append(discourse_parsing_predictions)

    def compute(self, documents: list[Document]) -> dict[str, Union[torch.Tensor, float]]:
        return {}

    @staticmethod
    def unique(x: torch.Tensor, dim: int = None):
        unique, inverse = torch.unique(x, sorted=True, return_inverse=True, dim=dim)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        return inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
