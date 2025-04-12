import random
from typing import Any

import pytest

from kwja.utils.sub_document import SequenceSplitter, SpanCandidate


def test_split_spans() -> None:
    splitter = SequenceSplitter(sequence_lengths=[20, 40, 30, 50, 20, 50], max_length=100, stride=1)
    actual_spans = list(splitter.split_into_spans())
    expected_spans = [
        (0, 3),  # [20, 40, 30]
        (2, 4),  # [30, 50]
        (2, 5),  # [30, 50, 20]
        (4, 6),  # [20, 50]
    ]
    for actual_span, expected_span in zip(actual_spans, expected_spans):
        assert isinstance(actual_span, SpanCandidate)
        assert (actual_span.start, actual_span.end) == expected_span


def test_split_spans_auto_stride() -> None:
    splitter = SequenceSplitter(sequence_lengths=[20, 40, 30, 50, 20, 50], max_length=100, stride=-1)
    actual_spans = list(splitter.split_into_spans())
    expected_spans = [
        (0, 3),  # [20, 40, 30]
        (3, 5),  # [50, 20]
        (5, 6),  # [50]
    ]
    for actual_span, expected_span in zip(actual_spans, expected_spans):
        assert isinstance(actual_span, SpanCandidate)
        assert (actual_span.start, actual_span.end) == expected_span


random.seed(0)


@pytest.mark.parametrize(
    "case",
    [
        {
            "sequence_lengths": [random.randint(1, 10) * 10 for _ in range(random.randint(1, 20))],
            "max_length": random.randint(10, 500),
            "stride": random.randint(1, 10),
        }
        for _ in range(999)
    ],
)
def test_split_conditions(case: dict[str, Any]) -> None:
    sequence_lengths: list[int] = case["sequence_lengths"]
    max_length: int = case["max_length"]
    stride: int = case["stride"]
    splitter = SequenceSplitter(sequence_lengths, max_length, stride)
    sequence_ids = list(range(len(sequence_lengths)))
    union_ids = set()
    for idx, item in enumerate(splitter.split_into_spans(return_candidates=True)):
        assert isinstance(item, tuple)
        span: SpanCandidate
        candidates: list[SpanCandidate]
        span, candidates = item
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


@pytest.mark.parametrize(
    "case",
    [
        {
            "sequence_lengths": [random.randint(1, 10) * 10 for _ in range(random.randint(1, 20))],
            "max_length": random.randint(10, 500),
        }
        for _ in range(99)
    ],
)
def test_split_conditions_auto_stride(case: dict[str, Any]) -> None:
    sequence_lengths: list[int] = case["sequence_lengths"]
    max_length: int = case["max_length"]
    splitter = SequenceSplitter(sequence_lengths, max_length, stride=-1)
    sequence_ids = list(range(len(sequence_lengths)))
    concatenated_ids = []
    for item in splitter.split_into_spans(return_candidates=True):
        assert isinstance(item, tuple)
        span: SpanCandidate
        candidates: list[SpanCandidate]
        span, candidates = item
        assert len(candidates) == 1
        assert span == candidates[0]
        assert 0 <= span.start < span.end <= len(sequence_lengths)  # start and end index are in valid range
        concatenated_ids.extend(sequence_ids[span.start : span.end])
        if span.length > max_length:
            # corner case: the length of a single item exceeds the max length
            assert span.end - span.start == 1
            continue
        if span.end == len(sequence_lengths):
            continue
        assert span.length + sequence_lengths[span.end] > max_length  # the length of each sub-sequence is maximized
    assert concatenated_ids == sequence_ids  # all sequence ids are covered without excess or deficiency
