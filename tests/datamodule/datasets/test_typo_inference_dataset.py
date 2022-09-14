from jula.datamodule.datasets.typo_inference_dataset import TypoInferenceDataset

tokenizer_kwargs = {"do_word_tokenize": False, "additional_special_tokens": ["<k>", "<d>", "<_>", "<dummy>"]}


def test_init():
    _ = TypoInferenceDataset(["テスト", "テスト"], tokenizer_kwargs=tokenizer_kwargs)


def test_len():
    dataset = TypoInferenceDataset(["テスト", "テスト"], tokenizer_kwargs=tokenizer_kwargs)
    assert len(dataset) == 2


def test_getitem():
    max_seq_length = 512
    dataset = TypoInferenceDataset(["テスト", "テスト"], tokenizer_kwargs=tokenizer_kwargs)
    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "attention_mask" in item
        assert item["input_ids"].shape == (max_seq_length,)
        assert item["attention_mask"].shape == (max_seq_length,)
