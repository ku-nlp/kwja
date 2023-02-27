from pathlib import Path

import pytest
from transformers import PreTrainedTokenizerBase

from kwja.datamodule.datasets.base import BaseDataset


def test_init(fixture_data_dir: Path, char_tokenizer: PreTrainedTokenizerBase):
    path = fixture_data_dir / "datasets" / "base_files"
    _ = BaseDataset(path, char_tokenizer, max_seq_length=256, document_split_stride=1)


def test_init_error(fixture_data_dir: Path, char_tokenizer: PreTrainedTokenizerBase):
    path = fixture_data_dir / "datasets" / "base_files"
    with pytest.raises(AssertionError):
        # no such file or directory
        _ = BaseDataset(path / "xxx", char_tokenizer, max_seq_length=256, document_split_stride=1)
    with pytest.raises(AssertionError):
        _ = BaseDataset(path / "0.knp", char_tokenizer, max_seq_length=256, document_split_stride=1)  # not a directory


def test_load_documents(fixture_data_dir: Path):
    path = fixture_data_dir / "datasets" / "base_files"
    _ = BaseDataset._load_documents(path)


def test_split_document(fixture_data_dir: Path, char_tokenizer: PreTrainedTokenizerBase):
    path = fixture_data_dir / "datasets" / "base_files"
    dataset: BaseDataset = BaseDataset(path, char_tokenizer, max_seq_length=15, document_split_stride=1)
    assert len(dataset.orig_documents) == 2
    assert [d.doc_id for d in dataset.orig_documents] == ["test-0", "test-1"]
    assert len(dataset.documents) == 3
    assert [d.doc_id for d in dataset.documents] == ["test-0-s1i0", "test-0-s1i1", "test-1"]


def test_split_document_overflow(fixture_data_dir: Path, char_tokenizer: PreTrainedTokenizerBase):
    path = fixture_data_dir / "datasets" / "base_files"
    dataset: BaseDataset = BaseDataset(path, char_tokenizer, max_seq_length=3, document_split_stride=1)
    assert len(dataset.orig_documents) == 2
    assert [d.doc_id for d in dataset.orig_documents] == ["test-0", "test-1"]
    assert len(dataset.documents) == 3
    assert [d.doc_id for d in dataset.documents] == ["test-0-s1i0", "test-0-s1i1", "test-1-s1i0"]
