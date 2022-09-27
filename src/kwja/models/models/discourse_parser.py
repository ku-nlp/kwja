import torch
import torch.nn as nn
from transformers import PretrainedConfig

from kwja.utils.constants import DISCOURSE_RELATIONS


class DiscourseParser(nn.Module):
    def __init__(self, pretrained_model_config: PretrainedConfig) -> None:
        super().__init__()
        hidden_size = pretrained_model_config.hidden_size

        self.dropout = nn.Dropout(0.0)

        self.l_src = nn.Linear(hidden_size, hidden_size)
        self.l_tgt = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, len(DISCOURSE_RELATIONS), bias=False)

    def forward(self, pooled_outputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_len, _ = pooled_outputs.size()
        h_src = self.l_src(self.dropout(pooled_outputs))  # (b, seq, hid)
        h_tgt = self.l_tgt(self.dropout(pooled_outputs))  # (b, seq, hid)
        h = torch.tanh(self.dropout(h_src.unsqueeze(2) + h_tgt.unsqueeze(1)))  # (b, seq, seq, hid)
        return self.out(h)  # (b, seq, seq, rel)
