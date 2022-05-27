from pathlib import Path

from jula.datamodule.datasets.word_dataset import WordDataset

here = Path(__file__).absolute().parent
path = here.joinpath("knp_files")


def test_init():
    _ = WordDataset(str(path))


def test_getitem():
    max_seq_length = 512
    dataset = WordDataset(str(path), max_seq_length=max_seq_length)
    item = dataset[0]
    assert isinstance(item, dict)
    assert "input_ids" in item
    assert "attention_mask" in item
    assert item["input_ids"].shape == (max_seq_length,)
    assert item["attention_mask"].shape == (max_seq_length,)
