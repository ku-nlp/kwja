from pathlib import Path

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from kwja.datamodule.datasets.base import BaseDataset


@pytest.fixture()
def path() -> Path:
    return Path(__file__).absolute().parent.parent.parent / "data" / "datasets" / "base_files"


@pytest.fixture()
def tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("nlp-waseda/roberta-base-japanese")


def test_init(path: Path, tokenizer: PreTrainedTokenizerBase):
    _ = BaseDataset(path, tokenizer, max_seq_length=256, document_split_stride=1)


def test_init_error(path: Path, tokenizer: PreTrainedTokenizerBase):
    with pytest.raises(AssertionError):
        # no such file or directory
        _ = BaseDataset(path / "xxx", tokenizer, max_seq_length=256, document_split_stride=1)
    with pytest.raises(AssertionError):
        _ = BaseDataset(path / "0.knp", tokenizer, max_seq_length=256, document_split_stride=1)  # not a directory


def test_load_documents(path: Path, tokenizer: PreTrainedTokenizerBase):
    _ = BaseDataset._load_documents(path)


def test_split_document(path: Path, tokenizer: PreTrainedTokenizerBase):
    dataset: BaseDataset = BaseDataset(path, tokenizer, max_seq_length=15, document_split_stride=1)
    assert len(dataset.orig_documents) == 2
    assert [d.doc_id for d in dataset.orig_documents] == ["test-0", "test-1"]
    assert len(dataset.documents) == 3
    assert [d.doc_id for d in dataset.documents] == ["test-0-s1i0", "test-0-s1i1", "test-1"]


def test_split_document_overflow(path: Path, tokenizer: PreTrainedTokenizerBase):
    dataset: BaseDataset = BaseDataset(path, tokenizer, max_seq_length=3, document_split_stride=1)
    assert len(dataset.orig_documents) == 2
    assert [d.doc_id for d in dataset.orig_documents] == ["test-0", "test-1"]
    assert len(dataset.documents) == 3
    assert [d.doc_id for d in dataset.documents] == ["test-0-s1i0", "test-0-s1i1", "test-1-s1i0"]
