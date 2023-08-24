import tempfile
from pathlib import Path
from textwrap import dedent

from transformers import PreTrainedTokenizerFast

from kwja.datamodule.datasets import Seq2SeqInferenceDataset


def test_init(seq2seq_tokenizer: PreTrainedTokenizerFast):
    _ = Seq2SeqInferenceDataset(seq2seq_tokenizer, max_src_length=128, max_tgt_length=512)


def test_len(seq2seq_tokenizer: PreTrainedTokenizerFast):
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

    dataset = Seq2SeqInferenceDataset(
        seq2seq_tokenizer,
        max_src_length=128,
        max_tgt_length=512,
        senter_file=Path(senter_file.name),
    )
    assert len(dataset) == 4


def test_getitem(seq2seq_tokenizer: PreTrainedTokenizerFast):
    max_src_length = 64
    max_tgt_length = 512
    texts = ["今日は晴れだ", "散歩に行こう", "今日は雨だ", "家でゆっくりしよう"]
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

    dataset = Seq2SeqInferenceDataset(
        seq2seq_tokenizer,
        max_src_length=max_src_length,
        max_tgt_length=max_tgt_length,
        senter_file=Path(senter_file.name),
    )
    for i in range(len(dataset)):
        feature = dataset[i]
        assert feature.example_ids == i
        assert feature.src_text == texts[i]
        assert len(feature.input_ids) == max_src_length
        assert len(feature.attention_mask) == max_src_length
        assert len(feature.seq2seq_labels) == 0
