import logging
import tempfile
from pathlib import Path
from textwrap import dedent

import pytest
from rhoknp import Sentence
from transformers import PreTrainedTokenizerFast

from kwja.datamodule.datamodule import token_dataclass_data_collator
from kwja.datamodule.datasets import Seq2SeqInferenceDataset
from kwja.utils.seq2seq_format import Seq2SeqFormatter


@pytest.mark.parametrize("empty_file", [False, True])
def test_init(
    seq2seq_tokenizer: PreTrainedTokenizerFast, tmp_path: Path, empty_file: bool, caplog: pytest.LogCaptureFixture
) -> None:
    max_src_length = 128
    max_tgt_length = 512
    juman_file = None
    if empty_file:
        juman_file = tmp_path / "empty.juman"
        juman_file.write_text("", encoding="utf-8")
    dataset = Seq2SeqInferenceDataset(seq2seq_tokenizer, max_src_length, max_tgt_length, juman_file=juman_file)
    assert len(dataset) == 0
    assert caplog.records == []


def test_all_sentences_skipped(
    seq2seq_tokenizer: PreTrainedTokenizerFast, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    juman_file = tmp_path / "input.juman"
    juman_file.write_text(
        "# S-ID:test-0\n" + "あ _ あ 未定義語 15 その他 1 * 0 * 0\n" * 100 + "EOS\n", encoding="utf-8"
    )
    dataset = Seq2SeqInferenceDataset(seq2seq_tokenizer, 32, 128, juman_file=juman_file)
    assert len(dataset) == 0
    assert caplog.record_tuples == [
        (
            "kwja.datamodule.datasets.seq2seq_inference",
            logging.WARNING,
            "Length of source sentence is too long: " + "あ" * 100,
        ),
        (
            "kwja.datamodule.datasets.seq2seq_inference",
            logging.ERROR,
            "No examples to process. Make sure any texts are given and they are not too long.",
        ),
    ]


def test_len(seq2seq_tokenizer: PreTrainedTokenizerFast) -> None:
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
    with tempfile.NamedTemporaryFile("wt", delete=False, encoding="utf-8") as juman_file:
        juman_file.write(juman_text)
        juman_file_path = Path(juman_file.name)

    try:
        dataset = Seq2SeqInferenceDataset(
            seq2seq_tokenizer,
            max_src_length,
            max_tgt_length,
            juman_file=juman_file_path,
        )
    finally:
        juman_file_path.unlink(missing_ok=True)
    assert len(dataset) == 4


def test_getitem(seq2seq_tokenizer: PreTrainedTokenizerFast) -> None:
    max_src_length = 128
    max_tgt_length = 512
    surfs = [
        ["今日", "は", "晴れ", "だ"],
        ["散歩", "に", "行こう"],
        ["今日", "は", "雨", "だ"],
        ["家", "で", "ゆっくり", "しよう"],
    ]
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
    with tempfile.NamedTemporaryFile("wt", delete=False, encoding="utf-8") as juman_file:
        juman_file.write(juman_text)
        juman_file_path = Path(juman_file.name)

    try:
        dataset = Seq2SeqInferenceDataset(
            seq2seq_tokenizer,
            max_src_length,
            max_tgt_length,
            juman_file=juman_file_path,
        )
    finally:
        juman_file_path.unlink(missing_ok=True)

    for i in range(len(dataset)):
        feature = dataset[i]
        assert feature.example_ids == i
        assert feature.surfs == surfs[i]
        assert len(feature.input_ids) == max_src_length
        assert len(feature.attention_mask) == max_src_length
        assert len(feature.seq2seq_labels) == 0


@pytest.mark.parametrize("length_offset", [-1, 0, 1])
def test_source_length_limit(
    seq2seq_tokenizer: PreTrainedTokenizerFast,
    tmp_path: Path,
    length_offset: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentence = Sentence.from_jumanpp(
        "# S-ID:test-0\n今日 _ 今日 未定義語 15 その他 1 * 0 * 0\n"
        "は _ は 未定義語 15 その他 1 * 0 * 0\n晴れ _ 晴れ 未定義語 15 その他 1 * 0 * 0\nEOS\n"
    )
    source_length = len(Seq2SeqFormatter(seq2seq_tokenizer).get_src_tokens(sentence)) + 1
    max_src_length = source_length + length_offset
    juman_file = tmp_path / "input.juman"
    juman_file.write_text(sentence.to_jumanpp(), encoding="utf-8")

    dataset = Seq2SeqInferenceDataset(seq2seq_tokenizer, max_src_length, 512, juman_file=juman_file)

    if length_offset < 0:
        assert len(dataset) == 0
        warning_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
        assert warning_messages == [f"Length of source sentence is too long: {sentence.text}"]
    else:
        assert len(dataset) == 1
        assert caplog.records == []
        assert len(dataset[0].input_ids) == max_src_length
        assert len(dataset[0].attention_mask) == max_src_length


def test_skipped_sentence_in_batch(
    seq2seq_tokenizer: PreTrainedTokenizerFast, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    juman_file = tmp_path / "input.juman"
    juman_file.write_text(
        "# S-ID:test-0\n今日 _ 今日 未定義語 15 その他 1 * 0 * 0\nEOS\n"
        "# S-ID:test-1\n" + "あ _ あ 未定義語 15 その他 1 * 0 * 0\n" * 100 + "EOS\n"
        "# S-ID:test-2\n明日 _ 明日 未定義語 15 その他 1 * 0 * 0\nEOS\n",
        encoding="utf-8",
    )
    dataset = Seq2SeqInferenceDataset(seq2seq_tokenizer, 32, 512, juman_file=juman_file)

    assert [example.sid for example in dataset.examples] == ["test-0", "test-2"]
    assert caplog.record_tuples == [
        (
            "kwja.datamodule.datasets.seq2seq_inference",
            logging.WARNING,
            "Length of source sentence is too long: " + "あ" * 100,
        )
    ]
    batch = token_dataclass_data_collator([dataset[0], dataset[1]])
    assert batch["example_ids"].tolist() == [0, 1]
    assert batch["input_ids"].shape[0] == 2
