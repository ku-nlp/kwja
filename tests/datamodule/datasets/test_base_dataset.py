from pathlib import Path

import pytest

from kwja.datamodule.datasets.base_dataset import BaseDataset

here = Path(__file__).absolute().parent
path = here.joinpath("knp_files")

base_dataset_kwargs = dict(
    document_split_stride=1,
    model_name_or_path="nlp-waseda/roberta-base-japanese",
    max_seq_length=128,
    tokenizer_kwargs={},
)


def test_init():
    _ = BaseDataset(path, **base_dataset_kwargs)


def test_init_error():
    with pytest.raises(AssertionError):
        _ = BaseDataset(path / "xxx", **base_dataset_kwargs)  # no such file or directory
    with pytest.raises(AssertionError):
        _ = BaseDataset(path / "000.knp", **base_dataset_kwargs)  # not a directory


def test_load_documents():
    _ = BaseDataset._load_documents(path)


def test_split_document():
    dataset = BaseDataset(path, **{**base_dataset_kwargs, "max_seq_length": 13})
    assert len(dataset.orig_documents) == 2
    assert [doc.doc_id for doc in dataset.orig_documents] == ["000", "1"]
    assert len(dataset.documents) == 3
    assert [doc.doc_id for doc in dataset.documents] == ["000-s1i0", "000-s1i1", "1"]


def test_split_document_overflow():
    dataset = BaseDataset(path, **{**base_dataset_kwargs, "max_seq_length": 3})
    assert len(dataset.orig_documents) == 2
    assert [doc.doc_id for doc in dataset.orig_documents] == ["000", "1"]
    assert len(dataset.documents) == 3
    assert [doc.doc_id for doc in dataset.documents] == ["000-s1i0", "000-s1i1", "1-s1i0"]
