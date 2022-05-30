from pathlib import Path

import torch

from jula.datamodule.datasets.word_dataset import WordDataset

here = Path(__file__).absolute().parent
path = here.joinpath("knp_files")


def test_init():
    _ = WordDataset(str(path))


def test_getitem():
    max_seq_length = 512
    dataset = WordDataset(str(path), max_seq_length=max_seq_length)
    for i in range(len(dataset)):
        document = dataset.documents[i]
        item = dataset[i]
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "subword_map" in item
        assert item["input_ids"].shape == (max_seq_length,)
        assert item["attention_mask"].shape == (max_seq_length,)
        assert item["subword_map"].shape == (max_seq_length, max_seq_length)
        assert torch.sum(torch.sum(item["subword_map"], dim=1) != 0) == len(
            document.morphemes
        )
