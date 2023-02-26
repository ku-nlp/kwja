from typing import Tuple

import pytest
import torch

from kwja.modules.components.crf import CRF
from kwja.utils.constants import MASKED, NE_TAGS


def test_init() -> None:
    crf = CRF(NE_TAGS)

    assert crf.num_tags == len(NE_TAGS)
    assert crf.start_transitions.shape == (len(NE_TAGS),)
    assert crf.transitions.shape == (len(NE_TAGS), len(NE_TAGS))
    assert crf.end_transitions.shape == (len(NE_TAGS),)
    for i, source in enumerate(NE_TAGS):
        if source.startswith("I-"):
            assert crf.start_transitions[i].item() == MASKED
        for j, target in enumerate(NE_TAGS):
            if (
                (source.startswith("B-") or source.startswith("I-"))
                and target.startswith("I-")
                and source[2:] != target[2:]
            ):
                assert crf.transitions[i, j].item() == MASKED
            elif source == "O" and target.startswith("I-"):
                assert crf.transitions[i, j].item() == MASKED


@pytest.mark.parametrize(
    "batch_size, seq_length, reduction",
    [
        (2, 3, "token_mean"),
        (2, 3, "mean"),
        (2, 3, "sum"),
        (2, 3, "none"),
    ],
)
def test_forward(batch_size: int, seq_length: int, reduction: str) -> None:
    crf = CRF(NE_TAGS)

    emissions = torch.zeros((batch_size, seq_length, crf.num_tags), dtype=torch.float)
    tags = torch.zeros((batch_size, seq_length), dtype=torch.long)
    llh = crf(emissions, tags, reduction=reduction)
    if reduction == "token_mean":
        assert llh.shape == ()
    elif reduction == "mean":
        assert llh.shape == ()
    elif reduction == "sum":
        assert llh.shape == ()
    else:  # none
        assert llh.shape == (batch_size,)


@pytest.mark.parametrize(
    "batch_size, seq_length, target_span",
    [
        (2, 3, (0, 3)),
        (2, 3, (1, 2)),
    ],
)
def test_viterbi_decode(batch_size: int, seq_length: int, target_span: Tuple[int, int]) -> None:
    crf = CRF(NE_TAGS)

    crf.start_transitions.data = torch.zeros_like(crf.start_transitions)
    crf.transitions.data = torch.zeros_like(crf.transitions)
    crf.end_transitions.data = torch.zeros_like(crf.end_transitions)

    ne_index = [i for i, tag in enumerate(NE_TAGS) if tag.startswith("B-")][0]
    emissions = torch.zeros((batch_size, seq_length, crf.num_tags), dtype=torch.float)
    emissions[:, :, ne_index] = 1.0
    mask = torch.zeros((batch_size, seq_length), dtype=torch.long)
    mask[:, torch.arange(*target_span)] = 1
    decoded = crf.viterbi_decode(emissions, mask)
    assert decoded.shape == (batch_size, seq_length)
    assert (decoded[:, torch.arange(*target_span)] == ne_index).all()
    assert (decoded[:, torch.arange(target_span[0])] == NE_TAGS.index("O")).all()
    assert (decoded[:, torch.arange(target_span[1], seq_length)] == NE_TAGS.index("O")).all()
