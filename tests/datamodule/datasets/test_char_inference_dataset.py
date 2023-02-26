import pytest
from omegaconf import ListConfig
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from kwja.datamodule.datasets.char_inference_dataset import CharInferenceDataset


@pytest.fixture()
def tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("ku-nlp/roberta-base-japanese-char-wwm", do_word_tokenize=False)


def test_init(tokenizer: PreTrainedTokenizerBase):
    _ = CharInferenceDataset(ListConfig(["テスト", "テスト"]), tokenizer, max_seq_length=512)


def test_len(tokenizer: PreTrainedTokenizerBase):
    dataset = CharInferenceDataset(ListConfig(["テスト", "テスト"]), tokenizer, max_seq_length=512)
    assert len(dataset) == 2


def test_getitem(tokenizer: PreTrainedTokenizerBase):
    max_seq_length = 512
    dataset = CharInferenceDataset(ListConfig(["テスト", "テスト"]), tokenizer, max_seq_length)
    for i in range(len(dataset)):
        feature = dataset[i]
        assert feature.example_ids == i
        assert len(feature.input_ids) == max_seq_length
        assert len(feature.attention_mask) == max_seq_length
