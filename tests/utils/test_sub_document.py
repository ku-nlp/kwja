from kwja.datamodule.datasets.base_dataset import split_with_overlap


def test_chunk():
    sequence_lengths = [20, 40, 30, 50, 20, 50]
    actual_spans = split_with_overlap(sequence_lengths, max_length=100, stride=1)
    expected_spans = [
        (0, 3),  # [20, 40, 30]
        (2, 4),  # [30, 50]
        (2, 5),  # [30, 50, 20]
        (4, 6),  # [20, 50]
    ]
    assert list(actual_spans) == expected_spans
