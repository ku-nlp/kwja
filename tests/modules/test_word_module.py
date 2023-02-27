from pathlib import Path

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from kwja.datamodule.datamodule import dataclass_data_collator
from kwja.datamodule.datasets import WordDataset
from kwja.modules import WordModule

HPARAMS: DictConfig = OmegaConf.create(
    {
        "datamodule": {
            "valid": ["test"],
            "test": ["test"],
        },
        "encoder": {
            "_target_": "transformers.AutoModel.from_pretrained",
            "pretrained_model_name_or_path": "nlp-waseda/roberta-base-japanese",
            "add_pooling_layer": False,
        },
        "max_seq_length": 32,
        "training_tasks": [
            "reading_prediction",
            "morphological_analysis",
            "word_feature_tagging",
            "ner",
            "base_phrase_feature_tagging",
            "dependency_parsing",
            "cohesion_analysis",
            "discourse_parsing",
        ],
        "dependency_topk": 2,
        "cohesion_tasks": ["pas_analysis", "bridging_reference_resolution", "coreference_resolution"],
        "exophora_referents": ["著者", "読者", "不特定:人", "不特定:物"],
        "restrict_cohesion_target": True,
        "pas_cases": ["ガ", "ヲ", "ニ", "ガ２"],
        "br_cases": ["ノ"],
        "special_tokens": ["著者", "読者", "不特定:人", "不特定:物", "[NULL]", "[NA]", "[ROOT]"],
        "discourse_parsing_threshold": 0.0,
    }
)


def test_init() -> None:
    _ = WordModule(HPARAMS)


def test_steps(fixture_data_dir: Path, word_tokenizer: PreTrainedTokenizerBase) -> None:
    module = WordModule(HPARAMS)

    path = fixture_data_dir / "datasets" / "word_files"
    dataset = WordDataset(
        str(path),
        word_tokenizer,
        max_seq_length=HPARAMS.max_seq_length,
        document_split_stride=1,
        cohesion_tasks=ListConfig(HPARAMS.cohesion_tasks),
        exophora_referents=ListConfig(HPARAMS.exophora_referents),
        restrict_cohesion_target=HPARAMS.restrict_cohesion_target,
        pas_cases=ListConfig(HPARAMS.pas_cases),
        br_cases=ListConfig(HPARAMS.br_cases),
        special_tokens=ListConfig(HPARAMS.special_tokens),
    )
    data_loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        collate_fn=dataclass_data_collator,
    )

    for batch_idx, batch in enumerate(data_loader):
        loss = module.training_step(batch, batch_idx)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert module.validation_step(batch, batch_idx) is None
        assert module.test_step(batch, batch_idx) is None
        ret = module.predict_step(batch, batch_idx)
        assert "example_ids" in ret
        assert "reading_predictions" in ret
        assert "reading_subword_map" in ret
        assert "pos_logits" in ret
        assert "subpos_logits" in ret
        assert "conjtype_logits" in ret
        assert "conjform_logits" in ret
        assert "word_feature_probabilities" in ret
        assert "ne_predictions" in ret
        assert "base_phrase_feature_probabilities" in ret
        assert "dependency_predictions" in ret
        assert "dependency_type_predictions" in ret
        assert "cohesion_logits" in ret
        assert "discourse_predictions" in ret
