from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from kwja.datamodule.datamodule import dataclass_data_collator
from kwja.datamodule.datasets import TypoDataset
from kwja.modules import TypoModule

HPARAMS: DictConfig = OmegaConf.create(
    {
        "datamodule": {
            "valid": ["test"],
            "test": ["test"],
        },
        "encoder": {
            "_target_": "transformers.AutoModel.from_pretrained",
            "pretrained_model_name_or_path": "ku-nlp/roberta-base-japanese-char-wwm",
            "add_pooling_layer": False,
        },
        "max_seq_length": 32,
        "special_tokens": ["<k>", "<d>", "<_>", "<dummy>"],
    }
)


def test_init() -> None:
    _ = TypoModule(HPARAMS)


def test_steps(fixture_data_dir: Path, typo_tokenizer: PreTrainedTokenizerBase) -> None:
    module = TypoModule(HPARAMS)

    path = fixture_data_dir / "datasets" / "typo_files"
    dataset = TypoDataset(str(path), typo_tokenizer, HPARAMS.max_seq_length)
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
        assert "kdr_predictions" in ret
        assert "kdr_probabilities" in ret
        assert "ins_predictions" in ret
        assert "ins_probabilities" in ret
