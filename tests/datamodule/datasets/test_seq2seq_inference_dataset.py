import tempfile
from pathlib import Path
from textwrap import dedent

from transformers import PreTrainedTokenizerFast

from kwja.datamodule.datasets import Seq2SeqInferenceDataset


def test_init(seq2seq_tokenizer: PreTrainedTokenizerFast):
    max_src_length = 128
    max_tgt_length = 512
    _ = Seq2SeqInferenceDataset(seq2seq_tokenizer, max_src_length, max_tgt_length)


def test_len(seq2seq_tokenizer: PreTrainedTokenizerFast):
    max_src_length = 128
    max_tgt_length = 512
    juman_text = dedent(
        """\
        # S-ID:test-0-1
        今日 _ 今日 未定義語 15 その他 1 * 0 * 0
        は _ は 未定義語 15 その他 1 * 0 * 0
        晴れ _ 晴れ 未定義語 15 その他 1 * 0 * 0
        だ _ だ 未定義語 15 その他 1 * 0 * 0
        EOS
        # S-ID:test-0-2
        散歩 _ 散歩 未定義語 15 その他 1 * 0 * 0
        に _ に 未定義語 15 その他 1 * 0 * 0
        行こう _ 行こう 未定義語 15 その他 1 * 0 * 0
        EOS
        # S-ID:test-1-1
        今日 _ 今日 未定義語 15 その他 1 * 0 * 0
        は _ は 未定義語 15 その他 1 * 0 * 0
        雨 _ 雨 未定義語 15 その他 1 * 0 * 0
        だ _ だ 未定義語 15 その他 1 * 0 * 0
        EOS
        # S-ID:test-1-2
        家 _ 家 未定義語 15 その他 1 * 0 * 0
        で _ で 未定義語 15 その他 1 * 0 * 0
        ゆっくり _ ゆっくり 未定義語 15 その他 1 * 0 * 0
        しよう _ しよう 未定義語 15 その他 1 * 0 * 0
        EOS
        """
    )
    juman_file = tempfile.NamedTemporaryFile("wt")
    juman_file.write(juman_text)
    juman_file.seek(0)

    dataset = Seq2SeqInferenceDataset(
        seq2seq_tokenizer,
        max_src_length,
        max_tgt_length,
        juman_file=Path(juman_file.name),
    )
    assert len(dataset) == 4


def test_getitem(seq2seq_tokenizer: PreTrainedTokenizerFast):
    max_src_length = 128
    max_tgt_length = 512
    texts = ["今日は晴れだ", "散歩に行こう", "今日は雨だ", "家でゆっくりしよう"]
    juman_text = dedent(
        """\
        # S-ID:test-0-1
        今日 _ 今日 未定義語 15 その他 1 * 0 * 0
        は _ は 未定義語 15 その他 1 * 0 * 0
        晴れ _ 晴れ 未定義語 15 その他 1 * 0 * 0
        だ _ だ 未定義語 15 その他 1 * 0 * 0
        EOS
        # S-ID:test-0-2
        散歩 _ 散歩 未定義語 15 その他 1 * 0 * 0
        に _ に 未定義語 15 その他 1 * 0 * 0
        行こう _ 行こう 未定義語 15 その他 1 * 0 * 0
        EOS
        # S-ID:test-1-1
        今日 _ 今日 未定義語 15 その他 1 * 0 * 0
        は _ は 未定義語 15 その他 1 * 0 * 0
        雨 _ 雨 未定義語 15 その他 1 * 0 * 0
        だ _ だ 未定義語 15 その他 1 * 0 * 0
        EOS
        # S-ID:test-1-2
        家 _ 家 未定義語 15 その他 1 * 0 * 0
        で _ で 未定義語 15 その他 1 * 0 * 0
        ゆっくり _ ゆっくり 未定義語 15 その他 1 * 0 * 0
        しよう _ しよう 未定義語 15 その他 1 * 0 * 0
        EOS
        """
    )
    juman_file = tempfile.NamedTemporaryFile("wt")
    juman_file.write(juman_text)
    juman_file.seek(0)

    dataset = Seq2SeqInferenceDataset(
        seq2seq_tokenizer,
        max_src_length,
        max_tgt_length,
        juman_file=Path(juman_file.name),
    )
    for i in range(len(dataset)):
        feature = dataset[i]
        assert feature.example_ids == i
        assert feature.src_text == texts[i]
        assert len(feature.input_ids) == max_src_length
        assert len(feature.attention_mask) == max_src_length
        assert len(feature.seq2seq_labels) == 0
