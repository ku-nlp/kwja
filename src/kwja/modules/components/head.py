import math

import torch
from torch import nn

# The pairwise hidden state built by the word selection heads below is O(seq^2 * rel * hid):
# 1.8 GB for seq=256 with the base model, which dominates peak device memory during
# prediction. Building it in slices of the source dimension bounds that intermediate.
#
# The reduction that produces the logits runs over the hidden dimension, which is never
# split, so the arithmetic is unchanged -- only the order in which rows are filled in.
# The results are bit-identical at the dimensions the models actually use, on CPU and on
# CUDA alike. They can differ in the last bits at small dimensions, because the matrix
# multiplication picks its blocking from the operand shape; the tests cover that case and
# assert that it never moves an argmax.
_CHUNK_TARGET_BYTES = 32 * 1024 * 1024


def _source_chunk_size(source: torch.Tensor, chunking_allowed: bool) -> int:
    """Number of source positions to expand at once, or the full length to disable chunking.

    Callers disallow chunking while autograd is recording, because every slice would then be
    kept alive for the backward pass and there would be nothing to save, and while the module
    is in training mode, because dropout would draw its masks per slice rather than once for
    the whole pairwise tensor.
    """
    seq_length = source.size(1)
    if not chunking_allowed:
        return seq_length
    bytes_per_source_position = source[:, :1].numel() * seq_length * source.element_size()
    if bytes_per_source_position == 0:
        return seq_length
    return max(1, min(seq_length, _CHUNK_TARGET_BYTES // bytes_per_source_position))


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
        chunk_size = _source_chunk_size(h_source, chunking_allowed=not self.training and not torch.is_grad_enabled())
        if chunk_size >= h_source.size(1):
            hidden = self.dropout(self.activation(h_source.unsqueeze(2) + h_target.unsqueeze(1)))  # (b, seq, seq, hid)
            return self.output_layer(hidden)  # (b, seq, seq, label)
        return self._forward_chunked(h_source, h_target, chunk_size)

    def _forward_chunked(self, h_source: torch.Tensor, h_target: torch.Tensor, chunk_size: int) -> torch.Tensor:
        batch_size, seq_length = h_source.size(0), h_source.size(1)
        logits = torch.empty(
            (batch_size, seq_length, seq_length, self.output_layer.out_features),
            device=h_source.device,
            dtype=h_source.dtype,
        )
        unsqueezed_target = h_target.unsqueeze(1)  # (b, 1, seq, hid)
        for start in range(0, seq_length, chunk_size):
            stop = min(start + chunk_size, seq_length)
            hidden = self.dropout(
                self.activation(h_source[:, start:stop].unsqueeze(2) + unsqueezed_target)
            )  # (b, chunk, seq, hid)
            logits[:, start:stop] = self.output_layer(hidden)
        return logits


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
        chunk_size = _source_chunk_size(h_source, chunking_allowed=not self.training and not torch.is_grad_enabled())
        if chunk_size >= seq_length:
            hidden = self.dropout(
                self.activation(h_source.unsqueeze(2) + h_target.unsqueeze(1))
            )  # (b, seq, seq, rel, hid)
            return torch.einsum("bstlh,hl->bstl", hidden, self.classifier_weight)  # (b, seq, seq, rel)
        return self._forward_chunked(h_source, h_target, chunk_size)

    def _forward_chunked(self, h_source: torch.Tensor, h_target: torch.Tensor, chunk_size: int) -> torch.Tensor:
        batch_size, seq_length = h_source.size(0), h_source.size(1)
        num_relations = self.classifier_weight.size(1)
        logits = torch.empty(
            (batch_size, seq_length, seq_length, num_relations), device=h_source.device, dtype=h_source.dtype
        )
        unsqueezed_target = h_target.unsqueeze(1)  # (b, 1, seq, rel, hid)
        for start in range(0, seq_length, chunk_size):
            stop = min(start + chunk_size, seq_length)
            hidden = self.dropout(
                self.activation(h_source[:, start:stop].unsqueeze(2) + unsqueezed_target)
            )  # (b, chunk, seq, rel, hid)
            logits[:, start:stop] = torch.einsum("bstlh,hl->bstl", hidden, self.classifier_weight)
        return logits


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
        chunk_size = _source_chunk_size(source, chunking_allowed=not self.training and not torch.is_grad_enabled())
        if chunk_size >= source.size(1):
            hidden = self.dropout(self.activation(source.unsqueeze(2) + target.unsqueeze(1)))  # (b, seq, seq, rel, hid)
            return torch.einsum("bstlh,hl->bstl", hidden, self.classifier)  # (b, seq, seq, rel)
        return self._forward_chunked(source, target, chunk_size)

    def _forward_chunked(self, source: torch.Tensor, target: torch.Tensor, chunk_size: int) -> torch.Tensor:
        batch_size, seq_length = source.size(0), source.size(1)
        num_relations = self.classifier.size(1)
        logits = torch.empty(
            (batch_size, seq_length, seq_length, num_relations), device=source.device, dtype=source.dtype
        )
        unsqueezed_target = target.unsqueeze(1)  # (b, 1, seq, rel, hid)
        for start in range(0, seq_length, chunk_size):
            stop = min(start + chunk_size, seq_length)
            hidden = self.dropout(
                self.activation(source[:, start:stop].unsqueeze(2) + unsqueezed_target)
            )  # (b, chunk, seq, rel, hid)
            logits[:, start:stop] = torch.einsum("bstlh,hl->bstl", hidden, self.classifier)
        return logits


class LoRADelta(nn.Module):
    def __init__(self, num_labels: int, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.dense_a = nn.Parameter(torch.Tensor(hidden_size, rank, num_labels))
        self.dense_b = nn.Parameter(torch.Tensor(rank, hidden_size, num_labels))
        nn.init.kaiming_uniform_(self.dense_a, a=math.sqrt(5))
        nn.init.zeros_(self.dense_b)

    def forward(self) -> torch.Tensor:
        return torch.einsum("hrl,ril->hil", self.dense_a, self.dense_b)  # (hid, hid, label)
