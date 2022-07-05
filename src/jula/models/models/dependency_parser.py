from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from jula.utils.utils import DEPENDENCY_TYPES, IGNORE_INDEX


class DependencyParser(nn.Module):
    def __init__(self, pretrained_model_config: PretrainedConfig) -> None:
        super().__init__()
        hidden_size = pretrained_model_config.hidden_size
        # Dependency Parsing as Head Selection [Zhang+ EACL2017]
        self.W_a = nn.Linear(hidden_size, hidden_size)
        self.U_a = nn.Linear(hidden_size, hidden_size)
        self.v_a = nn.Linear(hidden_size, 1, bias=False)

        self.head = nn.Sequential(
            OrderedDict(
                [
                    ("dense", nn.Linear(hidden_size * 2, hidden_size)),
                    ("hidden_act", nn.Tanh()),
                    (
                        "dropout",
                        nn.Dropout(pretrained_model_config.hidden_dropout_prob),
                    ),
                    ("fc", nn.Linear(hidden_size, len(DEPENDENCY_TYPES))),
                ]
            )
        )

    def forward(
        self, pooled_outputs: torch.Tensor, dependencies: Optional[torch.Tensor] = None
    ):  # (batch_size, max_seq_len, hidden_size)
        h_i = self.W_a(pooled_outputs)
        h_j = self.U_a(pooled_outputs)
        dependency_logits = self.v_a(
            torch.tanh(h_i.unsqueeze(1) + h_j.unsqueeze(2))
        ).squeeze(-1)
        if dependencies is not None:
            dependencies = torch.where(
                dependencies != IGNORE_INDEX,
                dependencies,
                torch.zeros_like(dependencies),
            )
        else:
            dependencies = torch.argmax(dependency_logits, dim=2)

        batch_size, max_seq_len, hidden_size = pooled_outputs.shape
        # governor_embeddings[0][0][0] == pooled_outputs[0][dependencies[0][0][0]]
        # governor_embeddings[0][0][1] == pooled_outputs[0][dependencies[0][0][1]]
        # governor_embeddings[0][1][0] == pooled_outputs[0][dependencies[0][1][0]]
        # governor_embeddings[1][0][0] == pooled_outputs[1][dependencies[1][0][0]]
        governor_embeddings = torch.gather(
            pooled_outputs.unsqueeze(1).expand(
                batch_size, max_seq_len, max_seq_len, hidden_size
            ),
            dim=2,
            index=dependencies.unsqueeze(2)
            .unsqueeze(2)
            .expand(batch_size, max_seq_len, 1, hidden_size),
        )  # (batch_size, max_seq_len, TopK, hidden_size)
        dependency_type_logits = self.head(
            torch.cat([pooled_outputs, governor_embeddings.squeeze(2)], dim=2)
        )
        return dependency_logits, dependency_type_logits
