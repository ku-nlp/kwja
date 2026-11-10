import torch

from kwja.metrics.utils import first_occurrence_indices


def test_first_occurrence_indices() -> None:
    assert first_occurrence_indices(torch.tensor([2, 0, 2, 1, 0])).tolist() == [0, 1, 3]
    assert first_occurrence_indices(torch.tensor([], dtype=torch.long)).tolist() == []
