import pytest
from omegaconf import ListConfig
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from kwja.datamodule.datasets import TypoInferenceDataset


@pytest.fixture()
def tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(
        "ku-nlp/roberta-base-japanese-char-wwm",
        do_word_tokenize=False,
        additional_special_tokens=["<k>", "<d>", "<_>", "<dummy>"],
    )


def test_init(tokenizer: PreTrainedTokenizerBase):
    _ = TypoInferenceDataset(ListConfig(["テスト", "テスト"]), tokenizer, max_seq_length=256)


def test_len(tokenizer: PreTrainedTokenizerBase):
    dataset = TypoInferenceDataset(ListConfig(["テスト", "テスト"]), tokenizer, max_seq_length=256)
    assert len(dataset) == 2


def test_stash(tokenizer: PreTrainedTokenizerBase):
    dataset = TypoInferenceDataset(ListConfig(["テスト", "テスト…"]), tokenizer, max_seq_length=256)
    assert len(dataset) == 1
    assert len(dataset.stash) == 1


def test_getitem(tokenizer: PreTrainedTokenizerBase):
    max_seq_length = 256
    dataset = TypoInferenceDataset(ListConfig(["テスト", "テスト"]), tokenizer, max_seq_length=max_seq_length)
    for i in range(len(dataset)):
        feature = dataset[i]
        assert feature.example_ids == i
        assert len(feature.input_ids) == max_seq_length
        assert len(feature.attention_mask) == max_seq_length
