from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from kwja.datamodule.datamodule import token_dataclass_data_collator
from kwja.datamodule.datasets import TypoDataset
from kwja.modules import TypoModule

with initialize(version_base=None, config_path="../../configs"):
    cfg = compose(config_name="typo_module.debug", return_hydra_config=True, overrides=["max_seq_length=32"])
    HydraConfig.instance().set_config(cfg)
    OmegaConf.set_readonly(cfg.hydra, False)
    OmegaConf.resolve(cfg)


def test_init() -> None:
    _ = TypoModule(cfg)


def test_run(fixture_data_dir: Path) -> None:
    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=False,
        enable_checkpointing=False,
        devices=1,
        accelerator="cpu",
    )

    path = fixture_data_dir / "datasets" / "typo_files"
    typo_tokenizer = hydra.utils.instantiate(cfg.datamodule.train.jwtd.tokenizer)
    dataset = TypoDataset(str(path), typo_tokenizer, cfg.max_seq_length)
    data_loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=token_dataclass_data_collator)
    val_dataloaders = {"jwtd": data_loader}

    module = TypoModule(cfg)

    trainer.fit(model=module, train_dataloaders=data_loader, val_dataloaders=val_dataloaders)
    trainer.test(model=module, dataloaders=val_dataloaders)
    trainer.predict(model=module, dataloaders=val_dataloaders)
