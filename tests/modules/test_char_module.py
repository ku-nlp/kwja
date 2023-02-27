from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from kwja.datamodule.datamodule import dataclass_data_collator
from kwja.datamodule.datasets import CharDataset
from kwja.modules import CharModule

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
        "special_tokens": [],
    }
)


def test_init() -> None:
    _ = CharModule(HPARAMS)


def test_steps(fixture_data_dir: Path, char_tokenizer: PreTrainedTokenizerBase) -> None:
    module = CharModule(HPARAMS)

    path = fixture_data_dir / "datasets" / "char_files"
    dataset = CharDataset(str(path), char_tokenizer, HPARAMS.max_seq_length, denormalize_probability=0.0)
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
        assert "word_segmentation_predictions" in ret
        assert "word_norm_op_predictions" in ret
