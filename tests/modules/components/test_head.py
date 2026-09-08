from collections.abc import Callable

import pytest
import torch
from torch import nn

import kwja.modules.components.head as head_module
from kwja.modules.components.head import (
    LoRARelationWiseWordSelectionHead,
    LoRASequenceMultiLabelingHead,
    RelationWiseWordSelectionHead,
    SequenceLabelingHead,
    WordSelectionHead,
    _source_chunk_size,
)


@pytest.mark.parametrize(
    ("num_labels", "hidden_size", "hidden_dropout_prob"),
    [
        (2, 3, 0.0),
        (10, 2, 0.1),
    ],
)
def test_sequential_labeling_head(num_labels: int, hidden_size: int, hidden_dropout_prob: float) -> None:
    head = SequenceLabelingHead(num_labels, hidden_size, hidden_dropout_prob)
    batch_size, seq_length = 2, 5
    input_ = torch.ones(batch_size, seq_length, hidden_size)
    output = head(input_)
    assert output.size() == (batch_size, seq_length, num_labels)


@pytest.mark.parametrize(
    ("num_labels", "hidden_size", "hidden_dropout_prob", "rank"),
    [
        (2, 3, 0.0, 4),
        (10, 2, 0.1, 2),
    ],
)
def test_lora_sequential_multi_labeling_head(
    num_labels: int, hidden_size: int, hidden_dropout_prob: float, rank: bool
) -> None:
    head = LoRASequenceMultiLabelingHead(num_labels, hidden_size, hidden_dropout_prob, rank=rank)
    batch_size, seq_length = 2, 5
    input_ = torch.ones(batch_size, seq_length, hidden_size)
    output = head(input_)
    assert output.size() == (batch_size, seq_length, num_labels)


@pytest.mark.parametrize(
    ("num_labels", "hidden_size", "hidden_dropout_prob"),
    [
        (5, 3, 0.0),
        (1, 2, 0.1),
    ],
)
def test_word_selection_head(num_labels: int, hidden_size: int, hidden_dropout_prob: float) -> None:
    head = WordSelectionHead(num_labels, hidden_size, hidden_dropout_prob)
    batch_size, seq_length = 2, 5
    input_ = torch.ones(batch_size, seq_length, hidden_size)
    output = head(input_)
    assert output.size() == (batch_size, seq_length, seq_length, num_labels)


@pytest.mark.parametrize(
    ("num_labels", "hidden_size", "hidden_dropout_prob"),
    [
        (5, 3, 0.0),
        (1, 2, 0.1),
    ],
)
def test_relation_wise_word_selection_head(num_labels: int, hidden_size: int, hidden_dropout_prob: float) -> None:
    head = RelationWiseWordSelectionHead(num_labels, hidden_size, hidden_dropout_prob)
    batch_size, seq_length = 2, 5
    input_ = torch.ones(batch_size, seq_length, hidden_size)
    output = head(input_)
    assert output.size() == (batch_size, seq_length, seq_length, num_labels)


@pytest.mark.parametrize(
    ("num_relations", "hidden_size", "hidden_dropout_prob", "rank"),
    [
        (5, 3, 0.0, 4),
        (1, 2, 0.1, 2),
    ],
)
def test_lora_relation_wise_word_selection_head(
    num_relations: int, hidden_size: int, hidden_dropout_prob: float, rank: int
) -> None:
    head = LoRARelationWiseWordSelectionHead(num_relations, hidden_size, hidden_dropout_prob, rank=rank)
    batch_size, seq_length = 2, 5
    input_ = torch.ones(batch_size, seq_length, hidden_size)
    output = head(input_)
    assert output.size() == (batch_size, seq_length, seq_length, num_relations)


def test_source_chunk_size_falls_back_to_the_full_length() -> None:
    source = torch.zeros(2, 16, 8)
    assert _source_chunk_size(source, chunking_allowed=False) == 16
    assert _source_chunk_size(source, chunking_allowed=True) == 16  # fits in the target budget


def test_source_chunk_size_is_at_least_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(head_module, "_CHUNK_TARGET_BYTES", 1)
    source = torch.zeros(2, 16, 8)
    assert _source_chunk_size(source, chunking_allowed=True) == 1


def test_source_chunk_size_falls_back_when_the_source_is_degenerate() -> None:
    # A source with no positions, or none to expand into, leaves nothing to divide by.
    assert _source_chunk_size(torch.zeros(2, 0, 8), chunking_allowed=True) == 0
    assert _source_chunk_size(torch.zeros(2, 16, 0), chunking_allowed=True) == 16


@pytest.mark.parametrize(
    "build_head",
    [
        lambda: WordSelectionHead(3, 8, 0.1),
        lambda: RelationWiseWordSelectionHead(3, 8, 0.1),
        lambda: LoRARelationWiseWordSelectionHead(3, 8, 0.1, rank=2),
    ],
    ids=["word_selection", "relation_wise_word_selection", "lora_relation_wise_word_selection"],
)
def test_chunked_forward_matches_the_unchunked_one(
    build_head: Callable[[], nn.Module], monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    head = build_head().eval()
    input_ = torch.randn(2, 16, 8)
    with torch.no_grad():
        expected = head(input_)
        monkeypatch.setattr(head_module, "_CHUNK_TARGET_BYTES", 512)
        actual = head(input_)
    # Chunking splits the source positions, never the hidden dimension the logits are reduced
    # over, so the arithmetic is the same. The results can still differ in the last bits
    # because the matrix multiplication picks its blocking from the operand shape.
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))


@pytest.mark.parametrize(
    "build_head",
    [
        lambda: WordSelectionHead(3, 8, 0.1),
        lambda: RelationWiseWordSelectionHead(3, 8, 0.1),
        lambda: LoRARelationWiseWordSelectionHead(3, 8, 0.1, rank=2),
    ],
    ids=["word_selection", "relation_wise_word_selection", "lora_relation_wise_word_selection"],
)
def test_training_and_autograd_keep_the_unchunked_path(
    build_head: Callable[[], nn.Module], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(head_module, "_CHUNK_TARGET_BYTES", 1)
    head = build_head()
    input_ = torch.randn(2, 16, 8)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the chunked path must not run here")

    monkeypatch.setattr(type(head), "_forward_chunked", fail)

    head.train()
    with torch.no_grad():
        head(input_)  # dropout would draw one mask per slice

    head.eval()
    assert head(input_).requires_grad is True  # autograd would keep every slice alive
