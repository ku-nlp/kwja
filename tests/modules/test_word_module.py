from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import ListConfig, OmegaConf
from torch.utils.data import DataLoader

from kwja.datamodule.datamodule import word_dataclass_data_collator
from kwja.datamodule.datasets import WordDataset
from kwja.modules import WordModule

with initialize(version_base=None, config_path="../../configs"):
    cfg = compose(config_name="word_module.debug", return_hydra_config=True, overrides=["max_seq_length=32"])
    HydraConfig.instance().set_config(cfg)
    OmegaConf.set_readonly(cfg.hydra, False)
    OmegaConf.register_new_resolver("concat", lambda x, y: x + y, replace=True)
    OmegaConf.resolve(cfg)


def test_init() -> None:
    _ = WordModule(cfg)


def test_steps(fixture_data_dir: Path) -> None:
    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=False,
        enable_checkpointing=False,
        devices=1,
        accelerator="cpu",
    )

    path = fixture_data_dir / "datasets" / "word_files"
    word_tokenizer = hydra.utils.instantiate(cfg.datamodule.train.kyoto.tokenizer)
    dataset = WordDataset(
        str(path),
        word_tokenizer,
        max_seq_length=cfg.max_seq_length,
        document_split_stride=1,
        cohesion_tasks=ListConfig(cfg.cohesion_tasks),
        exophora_referents=ListConfig(cfg.exophora_referents),
        restrict_cohesion_target=cfg.restrict_cohesion_target,
        pas_cases=ListConfig(cfg.pas_cases),
        br_cases=ListConfig(cfg.br_cases),
        special_tokens=ListConfig(cfg.special_tokens),
    )
    data_loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=word_dataclass_data_collator)
    val_dataloaders = {"dummy": data_loader}

    cfg.datamodule.valid = {"dummy": ""}
    cfg.datamodule.test = {"dummy": ""}
    module = WordModule(cfg)

    trainer.fit(model=module, train_dataloaders=data_loader, val_dataloaders=val_dataloaders)
    trainer.test(model=module, dataloaders=val_dataloaders)
    trainer.predict(model=module, dataloaders=val_dataloaders)
