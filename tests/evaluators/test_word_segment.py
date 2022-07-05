import pytest

from jula.evaluators.word_segment import WordSegmenterMetric


@pytest.mark.parametrize(
    "ids, expected",
    [
        ([0, 1, 2], ["O", "B", "I"]),
    ],
)
def test_convert_ids_to_labels(ids: list[int], expected: list[str]) -> None:
    assert WordSegmenterMetric.convert_ids_to_labels(ids) == expected


@pytest.mark.parametrize(
    "preds, labels, expected_filtered_preds, expected_filtered_labels",
    [
        (
            ["O", "B", "I"],
            ["O", "B", "I"],
            ["B", "I"],
            ["B", "I"],
        ),
        (
            ["O", "B", "I"],
            ["O", "O", "O"],
            [],
            [],
        ),
        (
            ["O", "B", "I"],
            ["O", "B", "O"],
            ["B"],
            ["B"],
        ),
    ],
)
def test_filter_predictions(
    preds: list[str],
    labels: list[str],
    expected_filtered_preds: list[str],
    expected_filtered_labels: list[str],
):
    filtered_preds, filtered_labels = WordSegmenterMetric.filter_predictions(
        preds, labels
    )
    assert filtered_preds == expected_filtered_preds
    assert filtered_labels == expected_filtered_labels
