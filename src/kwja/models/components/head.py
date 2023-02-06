from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn


class SequenceLabelingHead(nn.Module):
    def __init__(
        self,
        num_labels: int,
        hidden_size: int,
        hidden_dropout_prob: float,
        multi_label: bool = False,
    ) -> None:
        super().__init__()
        modules = [
            ("dense", nn.Linear(hidden_size, hidden_size)),
            ("hidden_act", nn.GELU()),
            ("dropout", nn.Dropout(hidden_dropout_prob)),
            ("fc", nn.Linear(hidden_size, num_labels)),
        ]
        if multi_label:
            modules.append(("act", nn.Sigmoid()))
        self.output_layer = nn.Sequential(OrderedDict(modules))

    def forward(self, pooled: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.output_layer(pooled)
        return logits


class WordSelectionHead(nn.Module):
    def __init__(
        self,
        num_relations: int,
        hidden_size: int,
        hidden_dropout_prob: float,
    ) -> None:
        super().__init__()
        self.l_source = nn.Linear(hidden_size, hidden_size)
        self.l_target = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.output_layer = nn.Linear(hidden_size, num_relations, bias=False)

    def forward(self, pooled: torch.Tensor) -> Dict[str, torch.Tensor]:
        h_source = self.l_source(pooled)  # (b, seq, hid)
        h_target = self.l_target(pooled)  # (b, seq, hid)
        h = torch.tanh(h_source.unsqueeze(2) + h_target.unsqueeze(1))  # (b, seq, seq, hid)
        h = self.dropout(h)
        logits = self.output_layer(h)  # (b, seq, seq, rel)
        return logits
