from omegaconf import ListConfig
from transformers import PreTrainedTokenizerBase

from kwja.datamodule.datasets import Seq2SeqInferenceDataset


def test_init(seq2seq_tokenizer: PreTrainedTokenizerBase):
    _ = Seq2SeqInferenceDataset(
        ListConfig(["テスト", "サンプル"]),
        seq2seq_tokenizer,
        max_src_length=128,
        max_tgt_length=512,
    )


def test_len(seq2seq_tokenizer: PreTrainedTokenizerBase):
    dataset = Seq2SeqInferenceDataset(
        ListConfig(["テスト", "サンプル"]),
        seq2seq_tokenizer,
        max_src_length=128,
        max_tgt_length=512,
    )
    assert len(dataset) == 2


def test_getitem(seq2seq_tokenizer: PreTrainedTokenizerBase):
    max_src_length = 128
    texts = ["テスト", "サンプル"]
    dataset = Seq2SeqInferenceDataset(
        ListConfig(texts),
        seq2seq_tokenizer,
        max_src_length=max_src_length,
        max_tgt_length=512,
    )
    for i in range(len(dataset)):
        feature = dataset[i]
        assert feature.example_ids == i
        assert feature.src_text == texts[i]
        assert len(feature.input_ids) == max_src_length
        assert len(feature.attention_mask) == max_src_length
        assert len(feature.seq2seq_labels) == 0
