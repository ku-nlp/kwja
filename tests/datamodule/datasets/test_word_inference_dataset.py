from jula.datamodule.datasets.word_inference_dataset import WordInferenceDataset

tokenizer_kwargs = {"additional_special_tokens": ["著者", "読者", "不特定:人", "不特定:物", "[NULL]", "[NA]", "[ROOT]"]}


def test_init():
    _ = WordInferenceDataset(["テスト", "テスト"], tokenizer_kwargs=tokenizer_kwargs)


def test_len():
    dataset = WordInferenceDataset(["テスト", "テスト"], tokenizer_kwargs=tokenizer_kwargs)
    assert len(dataset) == 2


def test_getitem():
    max_seq_length = 512
    dataset = WordInferenceDataset(["テスト", "テスト"], tokenizer_kwargs=tokenizer_kwargs)
    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "attention_mask" in item
        assert item["input_ids"].shape == (max_seq_length,)
        assert item["attention_mask"].shape == (max_seq_length,)
