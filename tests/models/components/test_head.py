import pytest
import torch

from kwja.models.components.head import SequenceLabelingHead, WordSelectionHead


@pytest.mark.parametrize(
    "num_labels, hidden_size, hidden_dropout_prob, multi_label",
    [
        (2, 3, 0.0, False),
        (2, 3, 0.0, True),
    ],
)
def test_sequential_labeling_head(
    num_labels: int,
    hidden_size: int,
    hidden_dropout_prob: float,
    multi_label: bool,
) -> None:
    head = SequenceLabelingHead(num_labels, hidden_size, hidden_dropout_prob, multi_label)
    batch_size = 2
    inp = torch.ones(batch_size, hidden_size)
    out = head(inp)
    assert out.size() == (batch_size, num_labels)


@pytest.mark.parametrize("num_relations, hidden_size, hidden_dropout_prob", [(2, 3, 0.0)])
def test_word_selection_head(num_relations: int, hidden_size: int, hidden_dropout_prob: float):
    head = WordSelectionHead(num_relations, hidden_size, hidden_dropout_prob)
    batch_size = 2
    seq_length = 5
    inp = torch.ones(batch_size, seq_length, hidden_size)
    out = head(inp)
    assert out.size() == (batch_size, seq_length, seq_length, num_relations)
