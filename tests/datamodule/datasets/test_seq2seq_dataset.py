from pathlib import Path
from typing import List

from transformers import PreTrainedTokenizerFast

from kwja.datamodule.datasets import Seq2SeqDataset
from kwja.utils.constants import CANON_TOKEN, IGNORE_INDEX, LEMMA_TOKEN, NO_CANON_TOKEN, READING_TOKEN, SURF_TOKEN


def test_init(data_dir: Path, seq2seq_tokenizer: PreTrainedTokenizerFast):
    path = data_dir / "datasets" / "word_files"
    _ = Seq2SeqDataset(str(path), seq2seq_tokenizer, max_src_length=128, max_tgt_length=512)


def test_getitem(data_dir: Path, seq2seq_tokenizer: PreTrainedTokenizerFast):
    path = data_dir / "datasets" / "word_files"
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


def test_encode(data_dir: Path, seq2seq_tokenizer: PreTrainedTokenizerFast):
    path = data_dir / "datasets" / "word_files"
    max_src_length = 128
    max_tgt_length = 512
    dataset = Seq2SeqDataset(
        str(path),
        seq2seq_tokenizer,
        max_src_length=max_src_length,
        max_tgt_length=max_tgt_length,
    )
    num_examples = len(dataset)

    expecteds: List[List[str]] = [
        [
            SURF_TOKEN,
            "太郎",
            READING_TOKEN,
            "た",
            "ろう",
            LEMMA_TOKEN,
            "太郎",
            CANON_TOKEN,
            "太郎",
            "/",
            "た",
            "ろう",
            SURF_TOKEN,
            "と",
            READING_TOKEN,
            "と",
            LEMMA_TOKEN,
            "と",
            CANON_TOKEN,
            NO_CANON_TOKEN,
            SURF_TOKEN,
            "次郎",
            READING_TOKEN,
            "じ",
            "ろう",
            LEMMA_TOKEN,
            "次郎",
            CANON_TOKEN,
            "次郎",
            "/",
            "じ",
            "ろう",
            SURF_TOKEN,
            "は",
            READING_TOKEN,
            "は",
            LEMMA_TOKEN,
            "は",
            CANON_TOKEN,
            NO_CANON_TOKEN,
            SURF_TOKEN,
            "よく",
            READING_TOKEN,
            "よく",
            LEMMA_TOKEN,
            "よく",
            CANON_TOKEN,
            NO_CANON_TOKEN,
            SURF_TOKEN,
            "けん",
            "か",
            READING_TOKEN,
            "けん",
            "か",
            LEMMA_TOKEN,
            "けん",
            "か",
            CANON_TOKEN,
            "喧",
            "嘩",
            "/",
            "けん",
            "か",
            SURF_TOKEN,
            "する",
            READING_TOKEN,
            "する",
            LEMMA_TOKEN,
            "する",
            CANON_TOKEN,
            NO_CANON_TOKEN,
            "</s>",
        ],
        [
            SURF_TOKEN,
            "辛い",
            READING_TOKEN,
            "から",
            "い",
            LEMMA_TOKEN,
            "辛い",
            CANON_TOKEN,
            NO_CANON_TOKEN,
            SURF_TOKEN,
            "ラーメン",
            READING_TOKEN,
            "ら",
            "ー",
            "めん",
            LEMMA_TOKEN,
            "ラーメン",
            CANON_TOKEN,
            NO_CANON_TOKEN,
            SURF_TOKEN,
            "が",
            READING_TOKEN,
            "が",
            LEMMA_TOKEN,
            "が",
            CANON_TOKEN,
            NO_CANON_TOKEN,
            SURF_TOKEN,
            "好きな",
            READING_TOKEN,
            "すき",
            "な",
            LEMMA_TOKEN,
            "好き",
            "だ",
            CANON_TOKEN,
            NO_CANON_TOKEN,
            SURF_TOKEN,
            "ので",
            READING_TOKEN,
            "ので",
            LEMMA_TOKEN,
            "のだ",
            CANON_TOKEN,
            NO_CANON_TOKEN,
            SURF_TOKEN,
            "頼",
            "み",
            READING_TOKEN,
            "た",
            "のみ",
            LEMMA_TOKEN,
            "頼む",
            CANON_TOKEN,
            NO_CANON_TOKEN,
            SURF_TOKEN,
            "ました",
            READING_TOKEN,
            "ました",
            LEMMA_TOKEN,
            "ます",
            CANON_TOKEN,
            NO_CANON_TOKEN,
            "</s>",
        ],
    ]
    for i in range(num_examples):
        feature = dataset[i]
        seq2seq_label_ids = [x for x in feature.seq2seq_labels if x != IGNORE_INDEX]
        seq2seq_labels = seq2seq_tokenizer.convert_ids_to_tokens(seq2seq_label_ids)
        assert seq2seq_labels == expecteds[i]
