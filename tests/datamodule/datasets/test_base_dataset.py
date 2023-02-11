from pathlib import Path

import pytest
from transformers import AutoTokenizer

from kwja.datamodule.datasets.base_dataset import BaseDataset

here = Path(__file__).absolute().parent
path = here.joinpath("knp_files")

tokenizer = AutoTokenizer.from_pretrained("nlp-waseda/roberta-base-japanese")


def test_init():
    _ = BaseDataset(path, tokenizer, 256, 1)


def test_init_error():
    with pytest.raises(AssertionError):
        _ = BaseDataset(path / "xxx", tokenizer, 256, 1)  # no such file or directory
    with pytest.raises(AssertionError):
        _ = BaseDataset(path / "000.knp", tokenizer, 256, 1)  # not a directory


def test_load_documents():
    _ = BaseDataset._load_documents(path)


def test_split_document():
    dataset = BaseDataset(path, tokenizer, 15, 1)
    assert len(dataset.orig_documents) == 2
    assert [d.doc_id for d in dataset.orig_documents] == ["000", "1"]
    assert len(dataset.documents) == 3
    assert [d.doc_id for d in dataset.documents] == ["000-s1i0", "000-s1i1", "1"]


def test_split_document_overflow():
    dataset = BaseDataset(path, tokenizer, 3, 1)  # max_seq_length=3
    assert len(dataset.orig_documents) == 2
    assert [d.doc_id for d in dataset.orig_documents] == ["000", "1"]
    assert len(dataset.documents) == 3
    assert [d.doc_id for d in dataset.documents] == ["000-s1i0", "000-s1i1", "1-s1i0"]
