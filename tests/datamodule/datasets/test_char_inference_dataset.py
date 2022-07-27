from jula.datamodule.datasets.char_inference_dataset import CharInferenceDataset


def test_init():
    _ = CharInferenceDataset(["テスト", "テスト"])


def test_len():
    dataset = CharInferenceDataset(["テスト", "テスト"])
    assert len(dataset) == 2


def test_getitem():
    max_seq_length = 512
    dataset = CharInferenceDataset(["テスト", "テスト"])
    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "attention_mask" in item
        assert item["input_ids"].shape == (max_seq_length,)
        assert item["attention_mask"].shape == (max_seq_length,)
