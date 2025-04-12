import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Any

import numpy as np
from transformers import PreTrainedTokenizerBase

from kwja.datamodule.datasets import WordInferenceDataset


def test_init(word_tokenizer: PreTrainedTokenizerBase, dataset_kwargs: dict[str, Any]):
    max_seq_length = 256
    document_split_stride = 1
    _ = WordInferenceDataset(word_tokenizer, max_seq_length, document_split_stride, **dataset_kwargs)


def test_len(word_tokenizer: PreTrainedTokenizerBase, dataset_kwargs: dict[str, Any]):
    max_seq_length = 256
    document_split_stride = 1
    juman_text = dedent(
        """\
        # S-ID:test-0-0
        今日 _ 今日 未定義語 15 その他 1 * 0 * 0
        は _ は 未定義語 15 その他 1 * 0 * 0
        晴れ _ 晴れ 未定義語 15 その他 1 * 0 * 0
        だ _ だ 未定義語 15 その他 1 * 0 * 0
        EOS
        """
    )
    juman_file = tempfile.NamedTemporaryFile("wt")
    juman_file.write(juman_text)
    juman_file.seek(0)

    dataset = WordInferenceDataset(
        word_tokenizer, max_seq_length, document_split_stride, juman_file=Path(juman_file.name), **dataset_kwargs
    )
    assert len(dataset) == 1


def test_len_multi_doc(word_tokenizer: PreTrainedTokenizerBase, dataset_kwargs: dict[str, Any]):
    max_seq_length = 256
    document_split_stride = 1
    juman_text = dedent(
        """\
        # S-ID:test-0-0
        今日 _ 今日 未定義語 15 その他 1 * 0 * 0
        は _ は 未定義語 15 その他 1 * 0 * 0
        晴れ _ 晴れ 未定義語 15 その他 1 * 0 * 0
        だ _ だ 未定義語 15 その他 1 * 0 * 0
        EOS
        # S-ID:test-1-0
        今日 _ 今日 未定義語 15 その他 1 * 0 * 0
        は _ は 未定義語 15 その他 1 * 0 * 0
        雨 _ 雨 未定義語 15 その他 1 * 0 * 0
        だ _ だ 未定義語 15 その他 1 * 0 * 0
        EOS
        """
    )
    juman_file = tempfile.NamedTemporaryFile("wt")
    juman_file.write(juman_text)
    juman_file.seek(0)

    dataset = WordInferenceDataset(
        word_tokenizer, max_seq_length, document_split_stride, juman_file=Path(juman_file.name), **dataset_kwargs
    )
    assert len(dataset) == 2


def test_getitem(word_tokenizer: PreTrainedTokenizerBase, dataset_kwargs: dict[str, Any]):
    max_seq_length = 256
    document_split_stride = 1
    juman_text = dedent(
        """\
        # S-ID:test-0-0
        今日 _ 今日 未定義語 15 その他 1 * 0 * 0
        は _ は 未定義語 15 その他 1 * 0 * 0
        晴れ _ 晴れ 未定義語 15 その他 1 * 0 * 0
        だ _ だ 未定義語 15 その他 1 * 0 * 0
        EOS
        """
    )
    juman_file = tempfile.NamedTemporaryFile("wt")
    juman_file.write(juman_text)
    juman_file.seek(0)

    dataset = WordInferenceDataset(
        word_tokenizer,
        max_seq_length,
        document_split_stride,
        juman_file=Path(juman_file.name),
        **dataset_kwargs,
    )
    num_cohesion_rels = len([r for rels in dataset.cohesion_task2rels.values() for r in rels])
    for i in range(len(dataset)):
        feature = dataset[i]
        assert feature.example_ids == i
        assert len(feature.input_ids) == max_seq_length
        assert len(feature.attention_mask) == max_seq_length
        assert len(feature.ne_mask) == max_seq_length
        assert np.array(feature.subword_map).shape == (max_seq_length, max_seq_length)
        assert np.array(feature.reading_subword_map).shape == (max_seq_length, max_seq_length)
        assert np.array(feature.dependency_mask).shape == (max_seq_length, max_seq_length)
        assert np.array(feature.cohesion_mask).shape == (num_cohesion_rels, max_seq_length, max_seq_length)
