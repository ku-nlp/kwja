import pytest
import torch

from kwja.modules.components.head import (
    LoRASequenceMultiLabelingHead,
    RelationWiseWordSelectionHead,
    SequenceLabelingHead,
    WordSelectionHead,
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
