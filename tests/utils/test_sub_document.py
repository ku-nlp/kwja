import random
from typing import Any, Dict, List

import pytest

from kwja.datamodule.datasets.base_dataset import SequenceSplitter, SpanCandidate

random.seed(0)
cases = [
    {
        "sequence_lengths": [random.randint(1, 10) * 10 for _ in range(random.randint(1, 20))],
        "max_length": random.randint(10, 500),
        "stride": random.randint(1, 10),
    }
    for _ in range(999)
]


def test_split_spans():
    splitter = SequenceSplitter(sequence_lengths=[20, 40, 30, 50, 20, 50], max_length=100, stride=1)
    actual_spans = list(splitter.split_with_overlap())
    expected_spans = [
        (0, 3),  # [20, 40, 30]
        (2, 4),  # [30, 50]
        (2, 5),  # [30, 50, 20]
        (4, 6),  # [20, 50]
    ]
    for actual_span, expected_span in zip(actual_spans, expected_spans):
        assert isinstance(actual_span, SpanCandidate)
        assert (actual_span.start, actual_span.end) == expected_span


@pytest.mark.parametrize("case", cases)
def test_split_conditions(case: Dict[str, Any]):
    sequence_lengths: List[int] = case["sequence_lengths"]
    max_length: int = case["max_length"]
    stride: int = case["stride"]
    splitter = SequenceSplitter(sequence_lengths, max_length, stride)
    sequence_ids = list(range(len(sequence_lengths)))
    union_ids = set()
    for idx, (span, candidates) in enumerate(splitter.split_with_overlap(return_candidates=True)):
        assert 0 <= span.start < span.end <= len(sequence_lengths)  # start and end index are in valid range
        union_ids.update(sequence_ids[span.start : span.end])
        if span.length > max_length:
            # corner case: the length of a single item exceeds the max length
            assert span.end - span.start == 1
            assert all(c.length > max_length for c in candidates)  # all the candidate lengths exceed the max length
            continue
        candidates = [c for c in candidates if c.length <= max_length]

        if idx > 0:
            if span.stride != stride:
                # corner case: the stride of a single item is not the desired stride
                assert all(c.stride != stride for c in candidates)  # no other sub-sequence has the desired stride
                continue
            candidates = [c for c in candidates if c.stride == stride]
        assert all(c.length <= span.length for c in candidates)  # the length of each sub-sequence is maximized
    assert union_ids == set(sequence_ids)  # all sequence ids are covered
