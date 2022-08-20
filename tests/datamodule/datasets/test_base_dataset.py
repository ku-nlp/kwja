from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from jula.datamodule.datasets.base_dataset import BaseDataset

here = Path(__file__).absolute().parent
path = here.joinpath("knp_files")

base_dataset_kwargs = dict(document_split_stride=1, model_name_or_path="nlp-waseda/roberta-base-japanese")


def test_init():
    _ = BaseDataset(str(path), **base_dataset_kwargs)


def test_init_error():
    with pytest.raises(AssertionError):
        _ = BaseDataset(str(path / "xxx"), **base_dataset_kwargs)  # no such file or directory
    with pytest.raises(AssertionError):
        _ = BaseDataset(str(path / "000.knp"), **base_dataset_kwargs)  # not a directory
    with TemporaryDirectory() as temporary_path:
        with pytest.raises(AssertionError):
            _ = BaseDataset(temporary_path, **base_dataset_kwargs)  # directory with not KNP file


def test_load_documents():
    _ = BaseDataset._load_documents(path)


def test_len():
    dataset = BaseDataset(str(path), **base_dataset_kwargs)
    assert len(dataset) == 2
