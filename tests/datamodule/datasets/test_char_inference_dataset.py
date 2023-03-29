import tempfile
from pathlib import Path
from textwrap import dedent

from transformers import PreTrainedTokenizerBase

from kwja.datamodule.datasets import CharInferenceDataset


def test_init(char_tokenizer: PreTrainedTokenizerBase):
    _ = CharInferenceDataset(char_tokenizer, max_seq_length=512, document_split_stride=1)


def test_len(char_tokenizer: PreTrainedTokenizerBase):
    senter_text = dedent(
        """\
        # S-ID:test-0-0
        今日は晴れだ
        # S-ID:test-0-1
        散歩に行こう
        # S-ID:test-1-0
        今日は雨だ
        # S-ID:test-1-1
        家でゆっくりしよう
        """
    )
    senter_file = tempfile.NamedTemporaryFile("wt")
    senter_file.write(senter_text)
    senter_file.seek(0)

    dataset = CharInferenceDataset(
        char_tokenizer, max_seq_length=512, document_split_stride=1, senter_file=Path(senter_file.name)
    )
    assert len(dataset) == 2


def test_getitem(char_tokenizer: PreTrainedTokenizerBase):
    max_seq_length = 64
    senter_text = dedent(
        """\
        # S-ID:test-0-0
        今日は晴れだ
        # S-ID:test-0-1
        散歩に行こう
        # S-ID:test-1-0
        今日は雨だ
        # S-ID:test-1-1
        家でゆっくりしよう
        """
    )
    senter_file = tempfile.NamedTemporaryFile("wt")
    senter_file.write(senter_text)
    senter_file.seek(0)
    dataset = CharInferenceDataset(
        char_tokenizer, max_seq_length=max_seq_length, document_split_stride=1, senter_file=Path(senter_file.name)
    )
    for i in range(len(dataset)):
        feature = dataset[i]
        assert feature.example_ids == i
        assert len(feature.input_ids) == max_seq_length
        assert len(feature.attention_mask) == max_seq_length
