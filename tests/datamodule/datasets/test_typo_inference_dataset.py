from omegaconf import ListConfig
from transformers import PreTrainedTokenizerBase

from kwja.datamodule.datasets import TypoInferenceDataset


def test_init(typo_tokenizer: PreTrainedTokenizerBase):
    _ = TypoInferenceDataset(ListConfig(["テスト", "サンプル"]), typo_tokenizer, max_seq_length=256)


def test_len(typo_tokenizer: PreTrainedTokenizerBase):
    dataset = TypoInferenceDataset(ListConfig(["テスト", "サンプル"]), typo_tokenizer, max_seq_length=256)
    assert len(dataset) == 2


def test_stash(typo_tokenizer: PreTrainedTokenizerBase):
    dataset = TypoInferenceDataset(ListConfig(["テスト", "サンプル…"]), typo_tokenizer, max_seq_length=256)
    assert len(dataset) == 1
    assert len(dataset.stash) == 1


def test_getitem(typo_tokenizer: PreTrainedTokenizerBase):
    max_seq_length = 256
    dataset = TypoInferenceDataset(ListConfig(["テスト", "サンプル"]), typo_tokenizer, max_seq_length=max_seq_length)
    for i in range(len(dataset)):
        feature = dataset[i]
        assert feature.example_ids == i
        assert len(feature.input_ids) == max_seq_length
        assert len(feature.attention_mask) == max_seq_length
