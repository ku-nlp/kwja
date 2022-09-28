from collections import OrderedDict
from math import sqrt
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from kwja.utils.constants import BASE_PHRASE_FEATURES, IGNORE_INDEX, NE_TAGS, WORD_FEATURES


class PhraseAnalyzer(nn.Module):
    def __init__(self, pretrained_model_config: PretrainedConfig) -> None:
        super().__init__()

        hidden_size = pretrained_model_config.hidden_size
        hidden_dropout_prob = pretrained_model_config.hidden_dropout_prob
        self.ne_head = nn.Sequential(
            OrderedDict(
                [
                    ("dense", nn.Linear(hidden_size, hidden_size)),
                    ("hidden_act", nn.GELU()),
                    (
                        "dropout",
                        nn.Dropout(hidden_dropout_prob),
                    ),
                    ("fc", nn.Linear(hidden_size, len(NE_TAGS))),
                ]
            )
        )
        self.crf = CRF(NE_TAGS, batch_first=True)
        self.word_feature_head = nn.Sequential(
            OrderedDict(
                [
                    ("dense", nn.Linear(hidden_size, hidden_size)),
                    ("hidden_act", nn.GELU()),
                    (
                        "dropout",
                        nn.Dropout(hidden_dropout_prob),
                    ),
                    ("fc", nn.Linear(hidden_size, len(WORD_FEATURES))),
                    ("act", nn.Sigmoid()),
                ]
            )
        )
        self.base_phrase_feature_head = nn.Sequential(
            OrderedDict(
                [
                    ("dense", nn.Linear(hidden_size, hidden_size)),
                    ("hidden_act", nn.GELU()),
                    (
                        "dropout",
                        nn.Dropout(hidden_dropout_prob),
                    ),
                    ("fc", nn.Linear(hidden_size, len(BASE_PHRASE_FEATURES))),
                    ("act", nn.Sigmoid()),
                ]
            )
        )

    def forward(self, pooled_outputs: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        ne_logits, emissions = self.compute_ne_logits(pooled_outputs, batch)
        outputs["ne_logits"] = ne_logits
        word_feature_logits = self.word_feature_head(pooled_outputs)  # (batch_size, seq_length, num_word_features)
        outputs["word_feature_logits"] = word_feature_logits  # (batch_size, seq_length, num_base_phrase_features)
        base_phrase_feature_logits = self.base_phrase_feature_head(pooled_outputs)
        outputs["base_phrase_feature_logits"] = base_phrase_feature_logits
        if "ne_tags" in batch:
            tags = torch.where(batch["target_mask"], batch["ne_tags"], NE_TAGS.index("O"))
            ner_loss = self.crf(emissions, tags, mask=batch["target_mask"]) * -1.0
            outputs["ner_loss"] = ner_loss
        if "word_features" in batch:
            word_feature_mask = batch["word_features"].ne(IGNORE_INDEX)
            input_ = word_feature_logits * word_feature_mask
            target = batch["word_features"] * word_feature_mask
            word_feature_loss = self.compute_loss(input_, target.float(), torch.sum(word_feature_mask[:, :, 0], dim=1))
            outputs["word_feature_loss"] = word_feature_loss
        if "base_phrase_features" in batch:
            base_phrase_feature_mask = batch["base_phrase_features"].ne(IGNORE_INDEX)
            input_ = base_phrase_feature_logits * base_phrase_feature_mask
            target = batch["base_phrase_features"] * base_phrase_feature_mask
            base_phrase_feature_loss = self.compute_loss(
                input_,
                target.float(),
                torch.sum(base_phrase_feature_mask[:, :, 0], dim=1),
            )
            outputs["base_phrase_feature_loss"] = base_phrase_feature_loss
        return outputs

    def compute_ne_logits(
        self, pooled_outputs: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ne_logits = self.ne_head(pooled_outputs)  # (batch_size, seq_length, num_tags)
        device, shape = map(lambda x: getattr(ne_logits, x), ["device", "shape"])
        non_target = torch.full_like(ne_logits, -1024.0)
        non_target[:, :, NE_TAGS.index("O")] = 0.0
        emissions = torch.where(batch["target_mask"].unsqueeze(2).expand(shape), ne_logits, non_target)
        decoded = torch.tensor(self.crf.decode(emissions), device=device).unsqueeze(2).expand(shape)
        arrange = torch.ones_like(decoded) * torch.arange(shape[-1], device=device)
        ne_logits = torch.where(decoded == arrange, torch.full_like(ne_logits, 1024.0), ne_logits)
        return ne_logits, emissions

    @staticmethod
    def compute_loss(
        input_: torch.Tensor,
        target: torch.Tensor,
        num_units: torch.Tensor,
    ) -> torch.Tensor:
        # (b, seq, num_features)
        losses = F.binary_cross_entropy(input=input_, target=target, reduction="none")
        # 各featureのlossの和
        losses = torch.sum(torch.sum(losses, dim=1), dim=1)  # (b, )
        # binary_cross_entropyはIGNORE_INDEXを渡せない
        loss = torch.mean(losses / num_units)
        return loss


# Copyright (c) 2017 Kemal Kurniawan <kemal@kkurniawan.com>
# Released under the MIT license
# https://github.com/kmkurn/pytorch-crf/blob/master/LICENSE.txt
class CRF(nn.Module):
    """Conditional random field.
    This module implements a conditional random field [1]_.
    The forward computation of this class computes the log likelihood of the given sequence of tags and emission score
    tensor.
    This class also has `~CRF.decode` method which finds the best tag sequence given an emission score tensor using
    `Viterbi algorithm`_.

    Args:
        tags (`~tuple[str, ...]`): Tuple of tags.
        batch_first (`~bool`): Whether the first dimension corresponds to the size of a minibatch.

    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size ``(num_tags, num_tags)``.

    .. [1] Lafferty, J., McCallum, A., Pereira, F.,
       "Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data",
       In Proc. ICML 2001
    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, tags: tuple[str, ...], batch_first: bool = True) -> None:
        num_tags = len(tags)
        assert num_tags > 0, f"invalid number of tags: {num_tags}"
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first

        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters(tags)

    def reset_parameters(self, tags: tuple[str, ...]) -> None:
        bound = sqrt(6 / self.num_tags)
        nn.init.uniform_(self.start_transitions, -bound, bound)
        nn.init.uniform_(self.end_transitions, -bound, bound)
        nn.init.uniform_(self.transitions, -bound, bound)
        with torch.no_grad():
            for i, from_ in enumerate(tags):
                if from_.startswith("I-"):
                    self.start_transitions[i] = -1024.0
                for j, to in enumerate(tags):
                    if (
                        (from_.startswith("B-") or from_.startswith("I-"))
                        and to.startswith("I-")
                        and from_[2:] != to[2:]
                    ):
                        self.transitions[i, j] = -1024.0
                    elif from_ == "O" and to.startswith("I-"):
                        self.transitions[i, j] = -1024.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_tags={self.num_tags})"

    def forward(
        self,
        emissions: torch.FloatTensor,
        tags: torch.LongTensor,
        mask: Optional[torch.ByteTensor] = None,
        reduction: str = "token_mean",
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.

        Args:
            emissions (`~torch.FloatTensor`): Emission score tensor of size ``(seq_length, batch_size, num_tags)`` or
                                              ``(batch_size, seq_length, num_tags)`` (if batch_first is True).
            tags (`~torch.LongTensor`): Sequence of tags tensor of size ``(seq_length, batch_size)`` or
                                        ``(batch_size, seq_length, num_tags)`` (if batch_first is True).
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)`` or
                                        ``(batch_size, seq_length)`` (if batch_first is True).
            reduction: Specifies  the reduction to apply to the output: ``none|sum|mean|token_mean``.
                       ``none``: no reduction will be applied.
                       ``sum``: the output will be summed over batches.
                       ``mean``: the output will be averaged over batches.
                       ``token_mean``: the output will be averaged over tokens.

        Returns:
            `~torch.Tensor`: The log likelihood.
                             This will have size ``(batch_size,)`` or ``()`` (if reduction is ``none``).
        """
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ("none", "sum", "mean", "token_mean"):
            raise ValueError(f"invalid reduction: {reduction}")
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        numerator = self._compute_score(emissions, tags, mask)  # (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)  # (batch_size,)
        llh = numerator - denominator  # (batch_size,)

        if reduction == "none":
            return llh
        if reduction == "sum":
            return llh.sum()
        if reduction == "mean":
            return llh.mean()
        assert reduction == "token_mean"
        return llh.sum() / mask.type_as(emissions).sum()

    def decode(self, emissions: torch.FloatTensor, mask: Optional[torch.ByteTensor] = None) -> list[list[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.FloatTensor`): Emission score tensor of size ``(seq_length, batch_size, num_tags)`` or
                                              ``(batch_size, seq_length, num_tags)`` (if batch_first is True).
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)`` or
                                        ``(batch_size, seq_length)`` (if batch_first is True).

        Returns:
            `~list[list[int]]`: list of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _validate(
        self,
        emissions: torch.FloatTensor,
        tags: Optional[torch.LongTensor] = None,
        mask: Optional[torch.ByteTensor] = None,
    ) -> None:
        if emissions.dim() != 3:
            raise ValueError(f"emissions must have dimension of 3, got {emissions.dim()}")
        if emissions.size(2) != self.num_tags:
            raise ValueError(f"expected last dimension of emissions is {self.num_tags}, got {emissions.size(2)}")

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    "the first two dimensions of emissions and tags must match, "
                    f"got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}"
                )

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    "the first two dimensions of emissions and mask must match, "
                    f"got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}"
                )
            # no_empty_seq = not self.batch_first and mask[0].all()
            # no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            # if not no_empty_seq and not no_empty_seq_bf:
            #     raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
        self, emissions: torch.FloatTensor, tags: torch.LongTensor, mask: torch.ByteTensor
    ) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        # mask: (seq_length, batch_size)
        assert mask.shape == tags.shape
        # assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.type_as(emissions)

        # Start transition score and first emission
        score = self.start_transitions[tags[0]]  # (batch_size, )
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]  # (batch_size, )

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]  # (batch_size, )

        # End transition score
        tags = tags.transpose(1, 0)
        mask = mask.transpose(1, 0)
        seq_ends = torch.amax(mask.long() * torch.arange(mask.shape[-1], device=mask.device), dim=1, keepdim=True)
        last_tags = torch.gather(tags, 1, seq_ends).squeeze(1)
        score += self.end_transitions[last_tags]  # (batch_size, )

        return score

    def _compute_normalizer(self, emissions: torch.FloatTensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        # assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of (batch_size, num_tags) where for each batch,
        # the j-th column stores the score that the first timestep has tag j
        score = self.start_transitions + emissions[0]  # (batch_size, num_tags)

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            broadcast_score = score.unsqueeze(2)  # (batch_size, num_tags, 1)

            # Broadcast emission score for every possible current tag
            broadcast_emissions = emissions[i].unsqueeze(1)  # (batch_size, 1, num_tags)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where for each sample,
            # entry at row i and column j stores the sum of scores of all possible tag sequences so far
            # that end with transitioning from tag i to tag j and emitting
            next_score = broadcast_score + self.transitions + broadcast_emissions  # (batch_size, num_tags, num_tags)

            # Sum over all possible current tags, but we're in score space, so a sum becomes a log-sum-exp:
            # for each sample, entry i stores the sum of scores of all possible tag sequences so far, that end in tag i
            next_score = torch.logsumexp(next_score, dim=1)  # (batch_size, num_tags)

            # Set score to the next score if this timestep is valid (mask == 1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)  # (batch_size, num_tags)

        # End transition score
        score += self.end_transitions  # (batch_size, num_tags)

        # Sum (log-sum-exp) over all possible tags
        return torch.logsumexp(score, dim=1)  # (batch_size,)

    def _viterbi_decode(self, emissions: torch.FloatTensor, mask: torch.ByteTensor) -> list[list[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        score = self.start_transitions + emissions[0]  # (batch_size, num_tags)
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends with tag j
        # history saves where the best tags candidate transitioned from;
        # this is used when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            broadcast_score = score.unsqueeze(2)  # (batch_size, num_tags, 1)

            # Broadcast emission score for every possible current tag
            broadcast_emission = emissions[i].unsqueeze(1)  # (batch_size, 1, num_tags)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where for each sample,
            # entry at row i and column j stores the score of the best tag sequence so far
            # that ends with transitioning from tag i to tag j and emitting
            next_score = broadcast_score + self.transitions + broadcast_emission  # (batch_size, num_tags, num_tags)

            # Find the maximum score over all possible current tag
            next_score, indices = next_score.max(dim=1)  # (batch_size, num_tags)

            # Set score to the next score if this timestep is valid (mask == 1) and
            # save the index that produces the next score
            score = torch.where(mask[i].unsqueeze(1), next_score, score)  # (batch_size, num_tags)
            history.append(indices)

        # End transition score
        score += self.end_transitions  # (batch_size, num_tags)

        # Now, compute the best path for each sample

        seq_ends = mask.long().sum(dim=0) - 1  # (batch_size,)
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag sequence, and
            # trace it back again, and so on
            for hist in reversed(history[: seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list
