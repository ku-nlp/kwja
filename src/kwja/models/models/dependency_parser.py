from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from kwja.utils.constants import DEPENDENCY_TYPE2INDEX, IGNORE_INDEX


class DependencyParser(nn.Module):
    def __init__(self, pretrained_model_config: PretrainedConfig, k: int) -> None:
        super().__init__()
        hidden_size = pretrained_model_config.hidden_size
        # Dependency Parsing as Head Selection [Zhang+ EACL2017]
        self.W_a = nn.Linear(hidden_size, hidden_size)
        self.U_a = nn.Linear(hidden_size, hidden_size)
        self.v_a = nn.Linear(hidden_size, 1, bias=False)

        self.dependency_type_head = nn.Sequential(
            OrderedDict(
                [
                    ("dense", nn.Linear(hidden_size * 2, hidden_size)),
                    ("hidden_act", nn.Tanh()),
                    (
                        "dropout",
                        nn.Dropout(pretrained_model_config.hidden_dropout_prob),
                    ),
                    ("fc", nn.Linear(hidden_size, len(DEPENDENCY_TYPE2INDEX))),
                ]
            )
        )

        self.k = k

    def forward(self, pooled_outputs: torch.Tensor, dependencies: Optional[torch.Tensor] = None):
        # (b, seq, h)
        h_i = self.W_a(pooled_outputs)
        h_j = self.U_a(pooled_outputs)
        dependency_logits = self.v_a(torch.tanh(h_i.unsqueeze(1) + h_j.unsqueeze(2))).squeeze(-1)

        pooled_outputs = pooled_outputs.unsqueeze(2)
        batch_size, sequence_len, k, hidden_size = pooled_outputs.shape
        if dependencies is not None:
            # gather_indexにIGNORE(=-100)を渡すことはできないので、0に上書き
            dependencies = (dependencies * dependencies.ne(IGNORE_INDEX)).unsqueeze(2).unsqueeze(3)
        else:
            dependencies = torch.topk(dependency_logits, self.k, dim=2).indices.unsqueeze(3)
            k = self.k

        # head_hidden_states[0][0][0] == pooled_outputs[0][dependencies[0][0][0]]
        # head_hidden_states[0][1][0] == pooled_outputs[0][dependencies[0][1][0]]
        # head_hidden_states[1][0][0] == pooled_outputs[1][dependencies[1][0][0]]
        head_hidden_states = torch.gather(
            pooled_outputs.expand(batch_size, sequence_len, sequence_len, hidden_size),
            dim=2,
            index=dependencies.expand(batch_size, sequence_len, k, hidden_size),
        )
        # (b, seq, 1 or k, num_dependency_types)
        dependency_type_logits = self.dependency_type_head(
            torch.cat(
                [
                    pooled_outputs.expand(batch_size, sequence_len, k, hidden_size),
                    head_hidden_states,
                ],
                dim=-1,
            )
        )
        return dependency_logits, dependency_type_logits
