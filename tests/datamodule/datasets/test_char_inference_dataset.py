from omegaconf import ListConfig
from transformers import PreTrainedTokenizerBase

from kwja.datamodule.datasets import CharInferenceDataset


def test_init(char_tokenizer: PreTrainedTokenizerBase):
    _ = CharInferenceDataset(ListConfig([]), char_tokenizer, max_seq_length=512, document_split_stride=1)


def test_len(char_tokenizer: PreTrainedTokenizerBase):
    texts = ListConfig(
        [
            "今日は晴れだ。散歩に行こう。",
            "今日は雨だ。家でゆっくりしよう。",
        ]
    )

    dataset = CharInferenceDataset(texts, char_tokenizer, max_seq_length=512, document_split_stride=1)
    assert len(dataset) == 2


def test_getitem(char_tokenizer: PreTrainedTokenizerBase):
    texts = ListConfig(
        [
            "今日は晴れだ。散歩に行こう。",
            "今日は雨だ。家でゆっくりしよう。",
        ]
    )
    max_seq_length = 64

    dataset = CharInferenceDataset(texts, char_tokenizer, max_seq_length=max_seq_length, document_split_stride=1)
    for i in range(len(dataset)):
        feature = dataset[i]
        assert feature.example_ids == i
        assert len(feature.input_ids) == max_seq_length
        assert len(feature.attention_mask) == max_seq_length
