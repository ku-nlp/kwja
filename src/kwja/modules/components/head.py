import math

import torch
import torch.nn as nn


class SequenceLabelingHead(nn.Sequential):
    def __init__(self, num_labels: int, hidden_size: int, hidden_dropout_prob: float) -> None:
        super().__init__(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(hidden_size, num_labels),
        )


class LoRASequenceMultiLabelingHead(nn.Module):
    """
    In multi-labeling tasks such as word feature tagging and base phrase feature tagging, rare labels are easily ignored
     during training, leading to a decrease in macro-F1. This module provides a low-rank adaptation layer for each label
     to encourage learning of rare labels.
    c.f. https://github.com/microsoft/LoRA
    """

    def __init__(self, num_labels: int, hidden_size: int, hidden_dropout_prob: float, rank: int = 4) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.delta = LoRADelta(num_labels, hidden_size, rank)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier_weight = nn.Parameter(torch.Tensor(hidden_size, num_labels))
        self.classifier_bias = nn.Parameter(torch.Tensor(num_labels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.classifier_weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.classifier_weight.size(0))
        nn.init.uniform_(self.classifier_bias, -bound, bound)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        dense_out = self.dense(pooled)  # (b, seq, hid)
        dense_delta = self.delta()  # (hid, hid, label)
        dense_delta_out = torch.einsum("bsh,hil->bsil", pooled, dense_delta)  # (b, seq, hid, label)
        hidden = self.dropout(self.activation(dense_out.unsqueeze(dim=3) + dense_delta_out))  # (b, seq, hid, label)
        # (b, seq, label), (1, 1, label) -> (b, seq, label)
        logits = torch.einsum("bshl,hl->bsl", hidden, self.classifier_weight) + self.classifier_bias.view(1, 1, -1)
        return torch.sigmoid(logits)  # (b, seq, label)


class WordSelectionHead(nn.Module):
    def __init__(self, num_labels: int, hidden_size: int, hidden_dropout_prob: float) -> None:
        super().__init__()
        self.l_source = nn.Linear(hidden_size, hidden_size)
        self.l_target = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.output_layer = nn.Linear(hidden_size, num_labels, bias=False)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        h_source = self.l_source(pooled)  # (b, seq, hid)
        h_target = self.l_target(pooled)  # (b, seq, hid)
        hidden = self.dropout(self.activation(h_source.unsqueeze(2) + h_target.unsqueeze(1)))  # (b, seq, seq, hid)
        return self.output_layer(hidden)  # (b, seq, seq, label)


class RelationWiseWordSelectionHead(nn.Module):
    def __init__(self, num_relations: int, hidden_size: int, hidden_dropout_prob: float) -> None:
        super().__init__()
        self.l_source = nn.Linear(hidden_size, hidden_size * num_relations)
        self.l_target = nn.Linear(hidden_size, hidden_size * num_relations)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier_weight = nn.Parameter(torch.Tensor(hidden_size, num_relations))
        nn.init.kaiming_uniform_(self.classifier_weight, a=math.sqrt(5))

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, hidden_size = pooled.size()
        h_source = self.l_source(pooled).view(batch_size, seq_length, -1, hidden_size)  # (b, seq, rel, hid)
        h_target = self.l_target(pooled).view(batch_size, seq_length, -1, hidden_size)  # (b, seq, rel, hid)
        hidden = self.dropout(self.activation(h_source.unsqueeze(2) + h_target.unsqueeze(1)))  # (b, seq, seq, rel, hid)
        return torch.einsum("bstlh,hl->bstl", hidden, self.classifier_weight)  # (b, seq, seq, rel)


class LoRARelationWiseWordSelectionHead(nn.Module):
    def __init__(self, num_relations: int, hidden_size: int, hidden_dropout_prob: float, rank: int = 4) -> None:
        super().__init__()
        self.l_source = nn.Linear(hidden_size, hidden_size)
        self.l_target = nn.Linear(hidden_size, hidden_size)
        self.delta_source = LoRADelta(num_relations, hidden_size, rank)
        self.delta_target = LoRADelta(num_relations, hidden_size, rank)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Parameter(torch.Tensor(hidden_size, num_relations))
        nn.init.kaiming_uniform_(self.classifier, a=math.sqrt(5))

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        h_source = self.l_source(pooled)  # (b, seq, hid)
        h_target = self.l_target(pooled)  # (b, seq, hid)
        delta_source_out = torch.einsum("bsh,hil->bsli", pooled, self.delta_source())  # (b, seq, rel, hid)
        delta_target_out = torch.einsum("bsh,hil->bsli", pooled, self.delta_target())  # (b, seq, rel, hid)
        source = h_source.unsqueeze(2) + delta_source_out  # (b, seq, rel, hid)
        target = h_target.unsqueeze(2) + delta_target_out  # (b, seq, rel, hid)
        hidden = self.dropout(self.activation(source.unsqueeze(2) + target.unsqueeze(1)))  # (b, seq, seq, rel, hid)
        return torch.einsum("bstlh,hl->bstl", hidden, self.classifier)  # (b, seq, seq, rel)


class LoRADelta(nn.Module):
    def __init__(self, num_labels: int, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.dense_a = nn.Parameter(torch.Tensor(hidden_size, rank, num_labels))
        self.dense_b = nn.Parameter(torch.Tensor(rank, hidden_size, num_labels))
        nn.init.kaiming_uniform_(self.dense_a, a=math.sqrt(5))
        nn.init.zeros_(self.dense_b)

    def forward(self) -> torch.Tensor:
        return torch.einsum("hrl,ril->hil", self.dense_a, self.dense_b)  # (hid, hid, label)
