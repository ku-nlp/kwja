from jula.datamodule.datasets.raw_text_dataset import RawTextDataset


def test_init():
    _ = RawTextDataset(["テスト", "テスト"])


def test_len():
    dataset = RawTextDataset(["テスト", "テスト"])
    assert len(dataset) == 2


def test_getitem():
    max_seq_length = 512
    dataset = RawTextDataset(["テスト", "テスト"])
    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "attention_mask" in item
        assert item["input_ids"].shape == (max_seq_length,)
        assert item["attention_mask"].shape == (max_seq_length,)
