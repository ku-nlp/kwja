import logging
from pathlib import Path

import pytest
from omegaconf import ListConfig
from transformers import PreTrainedTokenizerBase

from kwja.datamodule.datasets import CharInferenceDataset


@pytest.mark.parametrize("empty_file", [False, True])
def test_init(
    char_tokenizer: PreTrainedTokenizerBase, tmp_path: Path, empty_file: bool, caplog: pytest.LogCaptureFixture
) -> None:
    max_seq_length = 512
    raw_text_file = None
    if empty_file:
        raw_text_file = tmp_path / "empty.txt"
        raw_text_file.write_text("", encoding="utf-8")
    dataset = CharInferenceDataset(ListConfig([]), char_tokenizer, max_seq_length, raw_text_file=raw_text_file)
    assert len(dataset) == 0
    assert caplog.records == []


def test_all_text_skipped(char_tokenizer: PreTrainedTokenizerBase, caplog: pytest.LogCaptureFixture) -> None:
    text = "あ" * 100
    dataset = CharInferenceDataset(ListConfig([text]), char_tokenizer, 32)
    assert len(dataset) == 0
    assert caplog.record_tuples == [
        ("kwja.datamodule.datasets.char_inference", logging.WARNING, f"Length of sub document is too long: {text}"),
        (
            "kwja.datamodule.datasets.char_inference",
            logging.ERROR,
            "No examples to process. Make sure any texts are given and they are not too long.",
        ),
    ]


def test_len(char_tokenizer: PreTrainedTokenizerBase) -> None:
    texts = ListConfig(
        [
            "今日は晴れだ。散歩に行こう。",
            "今日は雨だ。家でゆっくりしよう。",
        ]
    )
    max_seq_length = 512
    dataset = CharInferenceDataset(texts, char_tokenizer, max_seq_length)
    assert len(dataset) == 2


def test_getitem(char_tokenizer: PreTrainedTokenizerBase) -> None:
    texts = ListConfig(
        [
            "今日は晴れだ。散歩に行こう。",
            "今日は雨だ。家でゆっくりしよう。",
        ]
    )
    max_seq_length = 512
    dataset = CharInferenceDataset(texts, char_tokenizer, max_seq_length)
    for i in range(len(dataset)):
        feature = dataset[i]
        assert feature.example_ids == i
        assert len(feature.input_ids) == max_seq_length
        assert len(feature.attention_mask) == max_seq_length


def test_skipped_documents(char_tokenizer: PreTrainedTokenizerBase, caplog: pytest.LogCaptureFixture) -> None:
    texts = ListConfig(["今日は晴れだ。" + "あ" * 100])
    max_seq_length = 32
    dataset = CharInferenceDataset(texts, char_tokenizer, max_seq_length)
    assert len(dataset) == 1
    assert [dataset.doc_id2document[example.doc_id].text for example in dataset.examples] == ["今日は晴れだ。"]
    assert caplog.record_tuples == [
        (
            "kwja.datamodule.datasets.char_inference",
            logging.WARNING,
            "Length of sub document is too long: " + "あ" * 100,
        ),
    ]
