from omegaconf import ListConfig
from transformers import AutoTokenizer

from kwja.datamodule.datasets.typo_inference_dataset import TypoInferenceDataset

tokenizer = AutoTokenizer.from_pretrained(
    "ku-nlp/roberta-base-japanese-char-wwm",
    do_word_tokenize=False,
    additional_special_tokens=["<k>", "<d>", "<_>", "<dummy>"],
)


def test_init():
    _ = TypoInferenceDataset(ListConfig(["テスト", "テスト"]), tokenizer, max_seq_length=256)


def test_len():
    dataset = TypoInferenceDataset(ListConfig(["テスト", "テスト"]), tokenizer, max_seq_length=256)
    assert len(dataset) == 2


def test_stash():
    dataset = TypoInferenceDataset(ListConfig(["テスト", "テスト", "テスト…"]), tokenizer, max_seq_length=256)
    assert len(dataset) == 2
    assert len(dataset.stash) == 1


def test_getitem():
    max_seq_length = 512
    dataset = TypoInferenceDataset(ListConfig(["テスト", "テスト"]), tokenizer, max_seq_length=max_seq_length)
    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "example_ids" in item
        assert "input_ids" in item
        assert "attention_mask" in item
        assert item["example_ids"] == i
        assert item["input_ids"].shape == (max_seq_length,)
        assert item["attention_mask"].shape == (max_seq_length,)
