from omegaconf import ListConfig

from kwja.datamodule.datasets.char_inference_dataset import CharInferenceDataset


def test_init():
    _ = CharInferenceDataset(ListConfig(["テスト", "テスト"]), document_split_stride=1)


def test_len():
    dataset = CharInferenceDataset(ListConfig(["テスト", "テスト"]), document_split_stride=1)
    assert len(dataset) == 2


def test_getitem():
    max_seq_length = 512
    dataset = CharInferenceDataset(ListConfig(["テスト", "テスト"]), document_split_stride=1)
    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "attention_mask" in item
        assert item["input_ids"].shape == (max_seq_length,)
        assert item["attention_mask"].shape == (max_seq_length,)
