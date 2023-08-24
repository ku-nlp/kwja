from pathlib import Path

import pytest
from transformers import PreTrainedTokenizerBase

from kwja.datamodule.datasets.base import FullAnnotatedDocumentLoaderMixin


def test_init(data_dir: Path, char_tokenizer: PreTrainedTokenizerBase):
    path = data_dir / "datasets" / "base_files"
    _ = FullAnnotatedDocumentLoaderMixin(path, char_tokenizer, max_seq_length=256, document_split_stride=1)


def test_init_error(data_dir: Path, char_tokenizer: PreTrainedTokenizerBase):
    path = data_dir / "datasets" / "base_files"
    with pytest.raises(AssertionError):
        # no such file or directory
        _ = FullAnnotatedDocumentLoaderMixin(path / "xxx", char_tokenizer, max_seq_length=256, document_split_stride=1)
    with pytest.raises(AssertionError):
        _ = FullAnnotatedDocumentLoaderMixin(
            path / "0.knp", char_tokenizer, max_seq_length=256, document_split_stride=1
        )  # not a directory


def test_load_documents(data_dir: Path):
    path = data_dir / "datasets" / "base_files"
    _ = FullAnnotatedDocumentLoaderMixin._load_documents(path, ext="knp")


def test_split_document(data_dir: Path, char_tokenizer: PreTrainedTokenizerBase):
    path = data_dir / "datasets" / "base_files"
    dataset = FullAnnotatedDocumentLoaderMixin(path, char_tokenizer, max_seq_length=15, document_split_stride=1)
    assert len(dataset.orig_documents) == 2
    assert [d.doc_id for d in dataset.orig_documents] == ["test-0", "test-1"]
    assert len(dataset.doc_id2document) == 3
    assert [doc_id for doc_id in dataset.doc_id2document] == ["test-0-s1i0", "test-0-s1i1", "test-1"]


def test_split_document_overflow(data_dir: Path, char_tokenizer: PreTrainedTokenizerBase):
    path = data_dir / "datasets" / "base_files"
    dataset = FullAnnotatedDocumentLoaderMixin(path, char_tokenizer, max_seq_length=3, document_split_stride=1)
    assert len(dataset.orig_documents) == 2
    assert [d.doc_id for d in dataset.orig_documents] == ["test-0", "test-1"]
    assert len(dataset.doc_id2document) == 3
    assert [doc_id for doc_id in dataset.doc_id2document] == ["test-0-s1i0", "test-0-s1i1", "test-1-s1i0"]
