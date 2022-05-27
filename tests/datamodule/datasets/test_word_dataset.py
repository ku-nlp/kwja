from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from jula.datamodule.datasets.word_dataset import WordDataset

here = Path(__file__).absolute().parent
path = here.joinpath("knp_files")


def test_init():
    _ = WordDataset(str(path))


def test_init_error():
    with pytest.raises(AssertionError):
        _ = WordDataset(str(path / "xxx"))  # no such file or directory
    with pytest.raises(AssertionError):
        _ = WordDataset(str(path / "000.knp"))  # not a directory
    with TemporaryDirectory() as temporary_path:
        with pytest.raises(AssertionError):
            _ = WordDataset(temporary_path)  # directory with not KNP file


def test_load_documents():
    _ = WordDataset.load_documents(path)


def test_len():
    dataset = WordDataset.load_documents(path)
    assert len(dataset) == 1
