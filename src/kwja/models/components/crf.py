from math import sqrt
from typing import Literal, Optional, Tuple

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
        reduction: Literal["token_mean", "sum", "mean", "none"] = "token_mean",
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(tags)

        numerator = self._compute_numerator(emissions, tags, mask)  # (b, )
        denominator = self._compute_denominator(emissions, mask)  # (b, )
        llh = denominator - numerator  # (b, )

        if reduction == "token_mean":
            return llh.sum() / mask.sum()
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
        heads = torch.amin(arange * mask + seq_len * (1 - mask), dim=1)
        head_tags = tags[indices, heads]
        min_head = int(heads.min())
        tails = torch.amax(arange * mask, dim=1)
        tail_tags = tags[indices, tails]
        max_tail = int(tails.max())

        score = self.start_transitions[head_tags] + emissions[indices, heads, head_tags]
        for j in range(min_head + 1, max_tail + 1):
            condition = torch.logical_and(mask[:, j] == 1, heads != j)
            next_score = score + self.transitions[tags[:, j - 1], tags[:, j]] + emissions[indices, j, tags[:, j]]
            score = torch.where(condition, next_score, score)
        score += self.end_transitions[tail_tags]

        return score  # (b, )

    # あり得るタグ系列のスコアの和を計算する
    def _compute_denominator(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, *_ = emissions.shape
        indices = torch.arange(batch_size)

        arange = torch.arange(seq_len, device=mask.device)
        heads = torch.amin(arange * mask + seq_len * (1 - mask), dim=1)
        min_head = int(heads.min())
        tails = torch.amax(arange * mask, dim=1)
        max_tail = int(tails.max())

        score = self.start_transitions + emissions[indices, heads]  # (b, num_tags)
        for j in range(min_head + 1, max_tail + 1):
            condition = torch.logical_and(mask[:, j] == 1, heads != j)
            broadcast_score = score.unsqueeze(2)  # (b, num_tags, 1)
            broadcast_emissions = emissions[:, j].unsqueeze(1)  # (b, 1, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions  # (b, num_tags, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)  # (b, num_tags)
            score = torch.where(condition.unsqueeze(1), next_score, score)
        score += self.end_transitions

        return torch.logsumexp(score, dim=1)  # (b, )

    def viterbi_decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = mask.shape
        indices = torch.arange(batch_size)

        arange = torch.arange(seq_len, device=mask.device)
        heads = torch.amin(arange * mask + seq_len * (1 - mask), dim=1)
        min_head = int(heads.min())
        tails = torch.amax(arange * mask, dim=1)
        max_tail = int(tails.max())

        score = self.start_transitions + emissions[indices, heads]  # (b, num_tags)
        history = []
        for j in range(min_head + 1, max_tail + 1):
            condition = torch.logical_and(mask[:, j] == 1, heads != j)
            broadcast_score = score.unsqueeze(2)  # (b, num_tags, 1)
            broadcast_emissions = emissions[:, j].unsqueeze(1)  # (b, 1, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions  # (b, num_tags, num_tags)
            next_score, max_indices = next_score.max(dim=1)  # (b, num_tags)
            score = torch.where(condition.unsqueeze(1), next_score, score)  # (b, num_tags)
            history.append(max_indices)
        score += self.end_transitions  # (b, num_tags)

        batch_best_tags = []
        for i in range(batch_size):
            head, tail = int(heads[i]), int(tails[i])
            _, best_tag = score[i].max(dim=0)
            best_tags = [best_tag.item()]
            span = slice(head - min_head, tail - min_head)
            for max_indices in history[span][::-1]:
                best_tag = max_indices[i][best_tags[-1]]
                best_tags.append(best_tag.item())
            best_tags += [self.tags.index("O")] * head
            best_tags = best_tags[::-1]
            best_tags += [self.tags.index("O")] * (seq_len - tail - 1)
            assert len(best_tags) == seq_len, "the length of decoded sequence is inconsistent with max seq length"
            batch_best_tags.append(best_tags)

        return torch.as_tensor(batch_best_tags, device=emissions.device)