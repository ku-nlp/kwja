from math import sqrt
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn

from kwja.utils.constants import MASKED


class CRF(nn.Module):
    def __init__(self, tags: Tuple[str, ...]) -> None:
        super().__init__()
        self.tags = tags
        self.num_tags = len(tags)

        self.start_transitions = nn.Parameter(torch.empty(self.num_tags))
        self.transitions = nn.Parameter(torch.empty(self.num_tags, self.num_tags))
        self.end_transitions = nn.Parameter(torch.empty(self.num_tags))

        self._initialize_parameters(tags)

    def _initialize_parameters(self, tags: Tuple[str, ...]) -> None:
        bound = sqrt(6 / self.num_tags)
        nn.init.uniform_(self.start_transitions, -bound, bound)
        nn.init.uniform_(self.transitions, -bound, bound)
        nn.init.uniform_(self.end_transitions, -bound, bound)
        with torch.no_grad():
            for i, source in enumerate(tags):
                if source.startswith("I-"):
                    self.start_transitions[i] = MASKED
                for j, target in enumerate(tags):
                    if (
                        (source.startswith("B-") or source.startswith("I-"))
                        and target.startswith("I-")
                        and source[2:] != target[2:]
                    ):
                        self.transitions[i, j] = MASKED
                    elif source == "O" and target.startswith("I-"):
                        self.transitions[i, j] = MASKED

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: Literal["token_mean", "mean", "sum", "none"] = "token_mean",
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.full_like(tags, True, dtype=torch.bool)

        numerator = self._compute_numerator(emissions, tags, mask)  # (b, )
        denominator = self._compute_denominator(emissions, mask)  # (b, )
        llh = denominator - numerator  # (b, )

        if reduction == "token_mean":
            labels_in_batch = mask.sum(dim=1)
            eps = 1e-6
            return (llh / (labels_in_batch + eps)).sum() / (labels_in_batch.ne(0).sum(0) + eps).sum()
        elif reduction == "mean":
            return llh.mean()
        elif reduction == "sum":
            return llh.sum()
        else:
            return llh

    # 正解のタグ系列のスコアを計算する
    def _compute_numerator(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = tags.shape
        indices = torch.arange(batch_size)

        arange = torch.arange(seq_len, device=mask.device)
        head_indices = torch.amin(torch.where(mask, arange, seq_len - 1), dim=1)
        head_tags = tags[indices, head_indices]
        min_head_index = int(head_indices.min())
        tail_indices = torch.amax(arange * mask, dim=1)
        tail_tags = tags[indices, tail_indices]
        max_tail_index = int(tail_indices.max())

        score: torch.Tensor = self.start_transitions[head_tags] + emissions[indices, head_indices, head_tags]
        for j in range(min_head_index + 1, max_tail_index + 1):
            condition = torch.logical_and(mask[:, j] == 1, head_indices != j)
            next_score = score + self.transitions[tags[:, j - 1], tags[:, j]] + emissions[indices, j, tags[:, j]]
            score = torch.where(condition, next_score, score)
        score = score + self.end_transitions[tail_tags]

        return torch.where(mask.sum(dim=1).ne(0), score, torch.zeros_like(score))  # (b, )

    # あり得るタグ系列のスコアの和を計算する
    def _compute_denominator(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, *_ = emissions.shape
        indices = torch.arange(batch_size)

        arange = torch.arange(seq_len, device=mask.device)
        head_indices = torch.amin(torch.where(mask, arange, seq_len - 1), dim=1)
        min_head_index = int(head_indices.min())
        tail_indices = torch.amax(arange * mask, dim=1)
        max_tail_index = int(tail_indices.max())

        score = self.start_transitions + emissions[indices, head_indices]  # (b, num_tags)
        for j in range(min_head_index + 1, max_tail_index + 1):
            condition = torch.logical_and(mask[:, j] == 1, head_indices != j)
            broadcast_score = score.unsqueeze(2)  # (b, num_tags, 1)
            broadcast_emissions = emissions[:, j].unsqueeze(1)  # (b, 1, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions  # (b, num_tags, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)  # (b, num_tags)
            score = torch.where(condition.unsqueeze(1), next_score, score)
        score = score + self.end_transitions
        score = torch.logsumexp(score, dim=1)  # (b, )

        return torch.where(mask.sum(dim=1).ne(0), score, torch.zeros_like(score))  # (b, )

    def viterbi_decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = mask.shape
        indices = torch.arange(batch_size)

        arange = torch.arange(seq_len, device=mask.device)
        head_indices = torch.amin(torch.where(mask, arange, seq_len - 1), dim=1)
        min_head_index = int(head_indices.min())
        tail_indices = torch.amax(arange * mask, dim=1)
        max_tail_index = int(tail_indices.max())

        score: torch.Tensor = self.start_transitions + emissions[indices, head_indices]  # (b, num_tags)
        history: List[torch.Tensor] = []
        for j in range(min_head_index + 1, max_tail_index + 1):
            condition = torch.logical_and(mask[:, j] == 1, head_indices != j)
            broadcast_score: torch.Tensor = score.unsqueeze(2)  # (b, num_tags, 1)
            broadcast_emissions: torch.Tensor = emissions[:, j].unsqueeze(1)  # (b, 1, num_tags)
            next_score: torch.Tensor = (
                broadcast_score + self.transitions + broadcast_emissions
            )  # (b, num_tags, num_tags)
            next_score, max_indices = next_score.max(dim=1)  # (b, num_tags)
            score = torch.where(condition.unsqueeze(1), next_score, score)  # (b, num_tags)
            history.append(max_indices)
        score += self.end_transitions  # (b, num_tags)

        batch_best_tags: List[List[int]] = []
        for i in range(batch_size):
            head_index, tail_index = int(head_indices[i]), int(tail_indices[i])
            _, best_tag = score[i].max(dim=0)
            assert isinstance(best_tag_int := best_tag.item(), int)
            best_tags: List[int] = [best_tag_int]
            for max_indices in history[head_index - min_head_index : tail_index - min_head_index][::-1]:
                best_tag = max_indices[i][best_tags[-1]]
                assert isinstance(best_tag_int := best_tag.item(), int)
                best_tags.append(best_tag_int)
            best_tags += [self.tags.index("O")] * head_index
            best_tags = best_tags[::-1]
            best_tags += [self.tags.index("O")] * (seq_len - tail_index - 1)
            assert len(best_tags) == seq_len, "the length of decoded sequence is inconsistent with max seq length"
            batch_best_tags.append(best_tags)

        return torch.as_tensor(batch_best_tags, device=emissions.device)
