from omegaconf import ListConfig

from jula.datamodule.datasets.word_inference_dataset import WordInferenceDataset

exophora_referents = ["著者", "読者", "不特定:人", "不特定:物"]
special_tokens = exophora_referents + ["[NULL]", "[NA]", "[ROOT]"]
word_dataset_kwargs = dict(
    pas_cases=ListConfig(["ガ", "ヲ", "ニ", "ガ２"]),
    bar_rels=ListConfig(["ノ"]),
    exophora_referents=ListConfig(exophora_referents),
    cohesion_tasks=ListConfig(["pas_analysis", "bridging", "coreference"]),
    special_tokens=ListConfig(special_tokens),
    tokenizer_kwargs={"additional_special_tokens": special_tokens},
)


def test_init():
    _ = WordInferenceDataset(["テスト", "テスト"], **word_dataset_kwargs)


def test_len():
    dataset = WordInferenceDataset(["テスト", "テスト"], **word_dataset_kwargs)
    assert len(dataset) == 2


def test_getitem():
    max_seq_length = 512
    dataset = WordInferenceDataset(["テスト", "テスト"], **word_dataset_kwargs)
    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "attention_mask" in item
        assert item["input_ids"].shape == (max_seq_length,)
        assert item["attention_mask"].shape == (max_seq_length,)
