from pathlib import Path

from transformers import PreTrainedTokenizerBase

from kwja.datamodule.datasets import Seq2SeqDataset
from kwja.utils.constants import IGNORE_INDEX


def test_init(fixture_data_dir: Path, seq2seq_tokenizer: PreTrainedTokenizerBase):
    path = fixture_data_dir / "datasets" / "word_files"
    _ = Seq2SeqDataset(str(path), seq2seq_tokenizer, max_src_length=128, max_tgt_length=512)


def test_getitem(fixture_data_dir: Path, seq2seq_tokenizer: PreTrainedTokenizerBase):
    path = fixture_data_dir / "datasets" / "word_files"
    max_src_length = 128
    max_tgt_length = 512
    dataset = Seq2SeqDataset(
        str(path),
        seq2seq_tokenizer,
        max_src_length=max_src_length,
        max_tgt_length=max_tgt_length,
    )
    for i in range(len(dataset)):
        feature = dataset[i]
        assert feature.example_ids == i
        assert len(feature.input_ids) == max_src_length
        assert len(feature.attention_mask) == max_src_length
        assert len(feature.seq2seq_labels) == max_tgt_length


def test_encode(fixture_data_dir: Path, seq2seq_tokenizer: PreTrainedTokenizerBase):
    path = fixture_data_dir / "datasets" / "word_files"
    max_src_length = 128
    max_tgt_length = 512
    dataset = Seq2SeqDataset(
        str(path),
        seq2seq_tokenizer,
        max_src_length=max_src_length,
        max_tgt_length=max_tgt_length,
    )
    num_examples = len(dataset)

    expecteds = [
        [
            "▁",
            "太郎",
            "▁",
            "た",
            "ろう",
            "▁",
            "太郎",
            "▁",
            "太郎",
            "/",
            "た",
            "ろう",
            "<extra_id_0>",  # 太郎
            "▁",
            "と",
            "▁",
            "と",
            "▁",
            "と",
            "▁",
            "<extra_id_2>",
            "<extra_id_0>",  # と
            "▁",
            "次郎",
            "▁",
            "じ",
            "ろう",
            "▁",
            "次郎",
            "▁",
            "次郎",
            "/",
            "じ",
            "ろう",
            "<extra_id_0>",  # 次郎
            "▁",
            "は",
            "▁",
            "は",
            "▁",
            "は",
            "▁",
            "<extra_id_2>",
            "<extra_id_0>",  # は
            "▁",
            "よく",
            "▁",
            "よく",
            "▁",
            "よく",
            "▁",
            "<extra_id_2>",
            "<extra_id_0>",  # よく
            "▁",
            "けん",
            "か",
            "▁",
            "けん",
            "か",
            "▁",
            "けん",
            "か",
            "▁",
            "喧",
            "嘩",
            "/",
            "けん",
            "か",
            "<extra_id_0>",  # けんか
            "▁",
            "する",
            "▁",
            "する",
            "▁",
            "する",
            "▁",
            "<extra_id_2>",
            "<extra_id_0>",  # する
            "</s>",
        ],
        [
            "▁",
            "辛い",
            "▁",
            "から",
            "い",
            "▁",
            "辛い",
            "▁",
            "<extra_id_2>",
            "<extra_id_0>",  # 辛い
            "▁",
            "ラーメン",
            "▁",
            "ら",
            "ー",
            "めん",
            "▁",
            "ラーメン",
            "▁",
            "<extra_id_2>",
            "<extra_id_0>",  # ラーメン
            "▁",
            "が",
            "▁",
            "が",
            "▁",
            "が",
            "▁",
            "<extra_id_2>",
            "<extra_id_0>",  # が
            "▁",
            "好きな",
            "▁",
            "すき",
            "な",
            "▁",
            "好き",
            "だ",
            "▁",
            "<extra_id_2>",
            "<extra_id_0>",  # 好きな
            "▁",
            "ので",
            "▁",
            "ので",
            "▁",
            "のだ",
            "▁",
            "<extra_id_2>",
            "<extra_id_0>",  # ので
            "▁",
            "頼",
            "み",
            "▁",
            "た",
            "のみ",
            "▁",
            "頼む",
            "▁",
            "<extra_id_2>",
            "<extra_id_0>",  # 頼み
            "▁",
            "ました",
            "▁",
            "ました",
            "▁",
            "ます",
            "▁",
            "<extra_id_2>",
            "<extra_id_0>",  # ました
            "</s>",
        ],
    ]
    for i in range(num_examples):
        feature = dataset[i]
        seq2seq_label_ids = [x for x in feature.seq2seq_labels if x != IGNORE_INDEX]
        seq2seq_labels = seq2seq_tokenizer.convert_ids_to_tokens(seq2seq_label_ids)
        assert seq2seq_labels == expecteds[i]
