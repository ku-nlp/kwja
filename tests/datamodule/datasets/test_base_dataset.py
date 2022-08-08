from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from jula.datamodule.datasets.base_dataset import BaseDataset

here = Path(__file__).absolute().parent
path = here.joinpath("knp_files")


def test_init():
    _ = BaseDataset(str(path), model_name_or_path="nlp-waseda/roberta-base-japanese")


def test_init_error():
    with pytest.raises(AssertionError):
        _ = BaseDataset(
            str(path / "xxx"), model_name_or_path="nlp-waseda/roberta-base-japanese"
        )  # no such file or directory
    with pytest.raises(AssertionError):
        _ = BaseDataset(str(path / "000.knp"), model_name_or_path="nlp-waseda/roberta-base-japanese")  # not a directory
    with TemporaryDirectory() as temporary_path:
        with pytest.raises(AssertionError):
            _ = BaseDataset(
                temporary_path, model_name_or_path="nlp-waseda/roberta-base-japanese"
            )  # directory with not KNP file


def test_load_documents():
    _ = BaseDataset.load_documents(path)


def test_len():
    dataset = BaseDataset(str(path), model_name_or_path="nlp-waseda/roberta-base-japanese")
    assert len(dataset) == 2
