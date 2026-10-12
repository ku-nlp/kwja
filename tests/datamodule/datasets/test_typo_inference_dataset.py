import logging

import pytest
from omegaconf import ListConfig
from transformers import PreTrainedTokenizerBase

from kwja.datamodule.datasets import TypoInferenceDataset


def test_init(typo_tokenizer: PreTrainedTokenizerBase) -> None:
    _ = TypoInferenceDataset(ListConfig(["テスト", "サンプル"]), typo_tokenizer, 512)


def test_len(typo_tokenizer: PreTrainedTokenizerBase) -> None:
    max_seq_length = 512
    dataset = TypoInferenceDataset(ListConfig(["テスト", "サンプル"]), typo_tokenizer, max_seq_length)
    assert len(dataset) == 2


def test_stash(typo_tokenizer: PreTrainedTokenizerBase) -> None:
    max_seq_length = 512
    dataset = TypoInferenceDataset(ListConfig(["テスト", "サンプル…"]), typo_tokenizer, max_seq_length)
    assert len(dataset) == 1
    assert len(dataset.stash) == 1


@pytest.mark.parametrize("text", ["あ" * 100, "サンプル…"])
def test_skipped_text(typo_tokenizer: PreTrainedTokenizerBase, text: str, caplog: pytest.LogCaptureFixture) -> None:
    dataset = TypoInferenceDataset(ListConfig([text]), typo_tokenizer, 32)
    assert len(dataset) == 1
    assert dataset.examples[0].pre_text == ""
    assert [stashed_text for stashed_text, _ in dataset.stash[0]] == [text]
    assert caplog.record_tuples == [
        (
            "kwja.datamodule.datasets.typo_inference",
            logging.WARNING,
            f"Skipping typo correction because text is too long or not tokenized character by character: {text}",
        )
    ]


def test_getitem(typo_tokenizer: PreTrainedTokenizerBase) -> None:
    max_seq_length = 512
    dataset = TypoInferenceDataset(ListConfig(["テスト", "サンプル"]), typo_tokenizer, max_seq_length)
    for i in range(len(dataset)):
        feature = dataset[i]
        assert feature.example_ids == i
        assert len(feature.input_ids) == max_seq_length
        assert len(feature.attention_mask) == max_seq_length
