import logging
import tempfile
from dataclasses import fields
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest
import torch
from transformers import PreTrainedTokenizerBase

from kwja.datamodule.datasets import WordInferenceDataset
from kwja.utils.sub_document import extract_target_sentences


@pytest.mark.parametrize("max_seq_length", [32, 256])
def test_encode_without_documents(
    word_tokenizer: PreTrainedTokenizerBase,
    dataset_kwargs: dict[str, Any],
    tmp_path: Path,
    max_seq_length: int,
) -> None:
    juman_file = tmp_path / "input.juman"
    juman_file.write_text(
        "".join(
            f"# S-ID:test-0-{i}\n" + "猫 _ 猫 名詞 6 普通名詞 1 * 0 * 0\n" * length + "EOS\n"
            for i, length in enumerate([1, 4, 3, 2, 5, 1, 4, 7])
        ),
        encoding="utf-8",
    )
    dataset = WordInferenceDataset(word_tokenizer, max_seq_length, 1, juman_file=juman_file, **dataset_kwargs)
    assert len(dataset) > 0
    assert (len(dataset) > 1) == (max_seq_length == 32)

    expected_features = [dataset[i] for i in range(len(dataset))]
    del dataset.doc_id2document
    del dataset.cohesion_task2extractor
    for index, expected in enumerate(expected_features):
        feature = dataset[index]
        for field in fields(feature):
            actual_value, expected_value = getattr(feature, field.name), getattr(expected, field.name)
            if isinstance(actual_value, torch.Tensor):
                assert torch.equal(actual_value, expected_value), field.name
            else:
                assert actual_value == expected_value, field.name


def test_masks_for_target_sentences(
    word_tokenizer: PreTrainedTokenizerBase,
    dataset_kwargs: dict[str, Any],
    tmp_path: Path,
) -> None:
    max_seq_length = 32
    juman_file = tmp_path / "input.juman"
    juman_file.write_text(
        "".join(f"# S-ID:test-0-{i}\n" + "猫 _ 猫 名詞 6 普通名詞 1 * 0 * 0\n" * 4 + "EOS\n" for i in range(8)),
        encoding="utf-8",
    )
    dataset = WordInferenceDataset(word_tokenizer, max_seq_length, 1, juman_file=juman_file, **dataset_kwargs)
    assert len(dataset) > 1
    example = dataset.examples[-1]
    document = dataset.doc_id2document[example.doc_id]
    target_morphemes = document.sentences[-1].morphemes
    start, stop = target_morphemes[0].global_index, target_morphemes[-1].global_index + 1
    assert start > 0
    feature = dataset[-1]
    assert feature.ne_mask == [start <= i < stop for i in range(max_seq_length)]
    assert not feature.dependency_mask[:start].any()
    assert not feature.dependency_mask[stop:].any()
    assert not feature.dependency_mask[start:stop, :start].any()
    assert feature.dependency_mask[start:stop].any(dim=1).all()
    assert not feature.cohesion_mask[:, :start].any()
    assert not feature.cohesion_mask[:, stop:].any()
    assert feature.cohesion_mask[:, start:stop, 0].all()


@pytest.mark.parametrize("empty_file", [False, True])
def test_init(
    word_tokenizer: PreTrainedTokenizerBase,
    dataset_kwargs: dict[str, Any],
    tmp_path: Path,
    empty_file: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    max_seq_length = 256
    document_split_stride = 1
    juman_file = None
    if empty_file:
        juman_file = tmp_path / "empty.juman"
        juman_file.write_text("", encoding="utf-8")
    dataset = WordInferenceDataset(
        word_tokenizer, max_seq_length, document_split_stride, juman_file=juman_file, **dataset_kwargs
    )
    assert len(dataset) == 0
    assert caplog.records == []


def test_all_sentences_skipped(
    word_tokenizer: PreTrainedTokenizerBase,
    dataset_kwargs: dict[str, Any],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    juman_file = tmp_path / "input.juman"
    juman_file.write_text("# S-ID:test-0\n" + "あ あ あ 感動詞 12 * 0 * 0 * 0\n" * 100 + "EOS\n", encoding="utf-8")
    dataset = WordInferenceDataset(word_tokenizer, 32, 1, juman_file=juman_file, **dataset_kwargs)
    assert len(dataset) == 0
    assert caplog.record_tuples == [
        (
            "kwja.datamodule.datasets.word_inference",
            logging.WARNING,
            "Length of sub document is too long: " + "あ" * 100,
        ),
        (
            "kwja.datamodule.datasets.word_inference",
            logging.ERROR,
            "No examples to process. Make sure any texts are given and they are not too long.",
        ),
    ]


def test_len(word_tokenizer: PreTrainedTokenizerBase, dataset_kwargs: dict[str, Any]) -> None:
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
    with tempfile.NamedTemporaryFile("wt", delete=False, encoding="utf-8") as juman_file:
        juman_file.write(juman_text)
        juman_file_path = Path(juman_file.name)

    try:
        dataset = WordInferenceDataset(
            word_tokenizer, max_seq_length, document_split_stride, juman_file=Path(juman_file.name), **dataset_kwargs
        )
    finally:
        juman_file_path.unlink(missing_ok=True)
    assert len(dataset) == 1


def test_len_multi_doc(word_tokenizer: PreTrainedTokenizerBase, dataset_kwargs: dict[str, Any]) -> None:
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
    with tempfile.NamedTemporaryFile("wt", delete=False, encoding="utf-8") as juman_file:
        juman_file.write(juman_text)
        juman_file_path = Path(juman_file.name)

    try:
        dataset = WordInferenceDataset(
            word_tokenizer, max_seq_length, document_split_stride, juman_file=Path(juman_file.name), **dataset_kwargs
        )
    finally:
        juman_file_path.unlink(missing_ok=True)
    assert len(dataset) == 2


def test_getitem(word_tokenizer: PreTrainedTokenizerBase, dataset_kwargs: dict[str, Any]) -> None:
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
    with tempfile.NamedTemporaryFile("wt", delete=False, encoding="utf-8") as juman_file:
        juman_file.write(juman_text)
        juman_file_path = Path(juman_file.name)

    try:
        dataset = WordInferenceDataset(
            word_tokenizer,
            max_seq_length,
            document_split_stride,
            juman_file=juman_file_path,
            **dataset_kwargs,
        )
    finally:
        juman_file_path.unlink(missing_ok=True)

    num_cohesion_rels = len([r for rels in dataset.cohesion_task2rels.values() for r in rels])
    for i in range(len(dataset)):
        feature = dataset[i]
        assert feature.example_ids == i
        assert len(feature.input_ids) == max_seq_length
        assert len(feature.attention_mask) == max_seq_length
        assert len(feature.ne_mask) == max_seq_length
        assert feature.subword_map.shape == (max_seq_length, max_seq_length)
        assert feature.reading_subword_map.shape == (max_seq_length, max_seq_length)
        assert feature.dependency_mask.shape == (max_seq_length, max_seq_length)
        assert feature.cohesion_mask.shape == (num_cohesion_rels, max_seq_length, max_seq_length)


def test_skipped_documents(
    word_tokenizer: PreTrainedTokenizerBase, dataset_kwargs: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    max_seq_length = 32
    document_split_stride = 1
    juman_text = (
        dedent(
            """\
            # S-ID:test-0-0
            今日 _ 今日 未定義語 15 その他 1 * 0 * 0
            は _ は 未定義語 15 その他 1 * 0 * 0
            晴れ _ 晴れ 未定義語 15 その他 1 * 0 * 0
            だ _ だ 未定義語 15 その他 1 * 0 * 0
            EOS
            # S-ID:test-0-1
            """
        )
        + "あ _ あ 未定義語 15 その他 1 * 0 * 0\n" * 40
        + dedent(
            """\
            EOS
            # S-ID:test-0-2
            明日 _ 明日 未定義語 15 その他 1 * 0 * 0
            は _ は 未定義語 15 その他 1 * 0 * 0
            雨 _ 雨 未定義語 15 その他 1 * 0 * 0
            だ _ だ 未定義語 15 その他 1 * 0 * 0
            EOS
            """
        )
    )
    with tempfile.NamedTemporaryFile("wt", delete=False, encoding="utf-8") as juman_file:
        juman_file.write(juman_text)
        juman_file_path = Path(juman_file.name)

    try:
        dataset = WordInferenceDataset(
            word_tokenizer, max_seq_length, document_split_stride, juman_file=juman_file_path, **dataset_kwargs
        )
    finally:
        juman_file_path.unlink(missing_ok=True)
    assert len(dataset) == 2
    assert [
        sentence.sid
        for example in dataset.examples
        for sentence in extract_target_sentences(dataset.doc_id2document[example.doc_id])
    ] == ["test-0-0", "test-0-2"]
    assert caplog.record_tuples == [
        (
            "kwja.datamodule.datasets.word_inference",
            logging.WARNING,
            "Length of sub document is too long: " + "あ" * 40,
        ),
    ]
