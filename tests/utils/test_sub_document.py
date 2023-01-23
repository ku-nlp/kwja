import random
from typing import Any, Dict, List

import pytest

from kwja.datamodule.datasets.base_dataset import split_with_overlap

random.seed(0)
cases = [
    {
        "sequence_lengths": [random.randint(1, 10) * 10 for _ in range(random.randint(1, 20))],
        "max_length": random.randint(10, 500),
        "stride": random.randint(1, 10),
    }
    for _ in range(100)
]


def test_split():
    sequence_lengths = [20, 40, 30, 50, 20, 50]
    actual_spans = split_with_overlap(sequence_lengths, max_length=100, stride=1)
    expected_spans = [
        (0, 3),  # [20, 40, 30]
        (2, 4),  # [30, 50]
        (2, 5),  # [30, 50, 20]
        (4, 6),  # [20, 50]
    ]
    assert list(actual_spans) == expected_spans


@pytest.mark.parametrize("case", cases)
def test_split_cases(case: Dict[str, Any]):
    sequence_lengths: List[int] = case["sequence_lengths"]
    max_length: int = case["max_length"]
    stride: int = case["stride"]
    actual_spans = split_with_overlap(sequence_lengths, max_length=max_length, stride=stride)
    sequence_ids = list(range(len(sequence_lengths)))
    union_ids = set()
    for start, end in actual_spans:
        assert 0 <= start < end <= len(sequence_lengths)  # start and end index are in valid range
        if end - start > 1:
            assert sum(sequence_lengths[start:end]) <= max_length  # sub-sequence length do not exceed max_length
        union_ids.update(sequence_ids[start:end])
    assert union_ids == set(sequence_ids)  # all sequence ids are covered
    # TODO: test stride and maximum context length
