from omegaconf import DictConfig, OmegaConf

import kwja.datamodule.datamodule as datamodule_module
from kwja.datamodule.datamodule import DataModule
from kwja.datamodule.datasets import CharInferenceDataset

# Any tokenizer works here; this one is small and already used by the other tests.
TOKENIZER_NAME = "ku-nlp/deberta-v2-tiny-japanese-char-wwm"


def _build_dataset_cfg() -> dict[str, object]:
    return {
        "_target_": "kwja.datamodule.datasets.CharInferenceDataset",
        "texts": ["今日は晴れだ。散歩に行こう。"],
        "max_seq_length": 512,
        "tokenizer": {
            "_target_": "transformers.AutoTokenizer.from_pretrained",
            "pretrained_model_name_or_path": TOKENIZER_NAME,
            "do_word_tokenize": False,
            "_convert_": "all",
        },
    }


def _build_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "train": {"corpus": _build_dataset_cfg()},
            "valid": {"corpus": _build_dataset_cfg()},
            "test": {"corpus": _build_dataset_cfg()},
            "predict": _build_dataset_cfg(),
            "batch_size": 1,
            "num_workers": 0,
            "dataset_type": "char",
        }
    )


def test_setup_predicting() -> None:
    datamodule_module._TOKENIZER_CACHE.clear()
    datamodule = DataModule(_build_cfg())
    datamodule.setup(stage="predict")
    assert datamodule.predict_dataset is not None
    assert len(datamodule.predict_dataset) == 1


def test_tokenizer_is_reused_across_datamodules() -> None:
    """A new DataModule is built for every prediction, so the tokenizer must be shared."""
    datamodule_module._TOKENIZER_CACHE.clear()
    cfg = _build_cfg()

    first = DataModule(cfg)
    first.setup(stage="predict")
    second = DataModule(cfg)
    second.setup(stage="predict")

    assert first.predict_dataset.tokenizer is second.predict_dataset.tokenizer


def test_tokenizer_is_reused_across_stages() -> None:
    datamodule_module._TOKENIZER_CACHE.clear()
    datamodule = DataModule(_build_cfg())

    datamodule.setup(stage="fit")
    assert datamodule.train_dataset is not None
    train_dataset = datamodule.train_dataset.datasets[0]
    valid_dataset = datamodule.valid_datasets["corpus"]

    datamodule.setup(stage="test")
    test_dataset = datamodule.test_datasets["corpus"]

    datamodule.setup(stage="predict")
    predict_dataset = datamodule.predict_dataset

    assert isinstance(train_dataset, CharInferenceDataset)
    assert isinstance(valid_dataset, CharInferenceDataset)
    assert isinstance(test_dataset, CharInferenceDataset)
    assert isinstance(predict_dataset, CharInferenceDataset)
    assert train_dataset.tokenizer is valid_dataset.tokenizer
    assert train_dataset.tokenizer is test_dataset.tokenizer
    assert train_dataset.tokenizer is predict_dataset.tokenizer


def test_different_tokenizer_configs_are_not_shared() -> None:
    """The cache is keyed by the resolved config, so a different config must not hit it."""
    datamodule_module._TOKENIZER_CACHE.clear()

    default = DataModule(_build_cfg())
    default.setup(stage="predict")

    modified = _build_cfg()
    modified.predict.tokenizer.additional_special_tokens = ["[NULL]"]
    other = DataModule(modified)
    other.setup(stage="predict")

    assert default.predict_dataset.tokenizer is not other.predict_dataset.tokenizer
