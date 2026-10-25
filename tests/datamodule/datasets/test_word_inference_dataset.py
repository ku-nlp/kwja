import logging
import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Any

import numpy as np
import pytest
from transformers import PreTrainedTokenizerBase

from kwja.datamodule.datasets import WordInferenceDataset
from kwja.utils.sub_document import extract_target_sentences


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
        assert np.array(feature.subword_map).shape == (max_seq_length, max_seq_length)
        assert np.array(feature.reading_subword_map).shape == (max_seq_length, max_seq_length)
        assert np.array(feature.dependency_mask).shape == (max_seq_length, max_seq_length)
        assert np.array(feature.cohesion_mask).shape == (num_cohesion_rels, max_seq_length, max_seq_length)


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


def test_cohesion_mask(word_tokenizer: PreTrainedTokenizerBase, dataset_kwargs: dict[str, Any], tmp_path: Path) -> None:
    max_seq_length = 20  # the document is split into two windows, the second with a context sentence
    document_split_stride = 1
    juman_file = tmp_path / "test.juman"
    juman_file.write_text(
        dedent(
            """\
            # S-ID:test-0-0
            今日 _ 今日 未定義語 15 その他 1 * 0 * 0
            は _ は 未定義語 15 その他 1 * 0 * 0
            晴れ _ 晴れ 未定義語 15 その他 1 * 0 * 0
            だ _ だ 未定義語 15 その他 1 * 0 * 0
            EOS
            # S-ID:test-0-1
            明日 _ 明日 未定義語 15 その他 1 * 0 * 0
            は _ は 未定義語 15 その他 1 * 0 * 0
            雨 _ 雨 未定義語 15 その他 1 * 0 * 0
            だ _ だ 未定義語 15 その他 1 * 0 * 0
            EOS
            # S-ID:test-0-2
            明後日 _ 明後日 未定義語 15 その他 1 * 0 * 0
            は _ は 未定義語 15 その他 1 * 0 * 0
            雪 _ 雪 未定義語 15 その他 1 * 0 * 0
            だ _ だ 未定義語 15 その他 1 * 0 * 0
            EOS
            """
        ),
        encoding="utf-8",
    )
    dataset = WordInferenceDataset(
        word_tokenizer, max_seq_length, document_split_stride, juman_file=juman_file, **dataset_kwargs
    )
    num_checked_rows = {"target": 0, "context": 0}
    for i in range(len(dataset)):
        example = dataset.examples[i]
        cohesion_mask = dataset[i].cohesion_mask  # (rel, seq, seq)
        morphemes = dataset.doc_id2document[example.doc_id].morphemes
        special_indices = set(example.special_token_indexer.get_morpheme_level_indices(only_cohesion=True))
        rel_index = 0
        for cohesion_task in dataset.cohesion_tasks:
            cohesion_extractor = dataset.cohesion_task2extractor[cohesion_task]
            for _ in dataset.cohesion_task2rels[cohesion_task]:
                for morpheme in morphemes:
                    row = cohesion_mask[rel_index, morpheme.global_index]
                    if morpheme.global_index in example.analysis_target_morpheme_indices:
                        candidates = cohesion_extractor.get_candidates(morpheme, morphemes)
                        assert (
                            set(row.nonzero().flatten().tolist())
                            == {c.global_index for c in candidates} | special_indices
                        )
                        num_checked_rows["target"] += len(candidates) > 0
                    else:
                        # the predictions of the context sentences are not used
                        assert not row.any()
                        num_checked_rows["context"] += 1
                rel_index += 1
        assert rel_index == len(cohesion_mask)
    # the windows hold both target rows with candidates and rows of context sentences
    assert num_checked_rows["target"] > 0
    assert num_checked_rows["context"] > 0
