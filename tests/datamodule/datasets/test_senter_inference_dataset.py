from omegaconf import ListConfig
from transformers import PreTrainedTokenizerBase

from kwja.datamodule.datasets import SenterInferenceDataset


def test_init(char_tokenizer: PreTrainedTokenizerBase):
    _ = SenterInferenceDataset(ListConfig(["テスト", "サンプル"]), char_tokenizer, max_seq_length=512)


def test_len(char_tokenizer: PreTrainedTokenizerBase):
    dataset = SenterInferenceDataset(ListConfig(["テスト", "サンプル"]), char_tokenizer, max_seq_length=512)
    assert len(dataset) == 2


def test_getitem(char_tokenizer: PreTrainedTokenizerBase):
    max_seq_length = 512
    dataset = SenterInferenceDataset(ListConfig(["テスト", "サンプル"]), char_tokenizer, max_seq_length)
    for i in range(len(dataset)):
        feature = dataset[i]
        assert feature.example_ids == i
        assert len(feature.input_ids) == max_seq_length
        assert len(feature.attention_mask) == max_seq_length
