from pathlib import Path

import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from kwja.datamodule.datamodule import dataclass_data_collator
from kwja.datamodule.datasets import SenterDataset
from kwja.modules import SenterModule

with initialize(version_base=None, config_path="../../configs"):
    cfg = compose(config_name="senter_module.debug", return_hydra_config=True, overrides=["max_seq_length=32"])
    HydraConfig.instance().set_config(cfg)
    OmegaConf.set_readonly(cfg.hydra, False)
    OmegaConf.resolve(cfg)


def test_init() -> None:
    _ = SenterModule(cfg)


def test_steps(fixture_data_dir: Path) -> None:
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=None,
        enable_checkpointing=False,
        devices=1,
        accelerator="cpu",
    )

    path = fixture_data_dir / "datasets" / "char_files"
    char_tokenizer = hydra.utils.instantiate(cfg.datamodule.train.kyoto.tokenizer)
    dataset = SenterDataset(str(path), char_tokenizer, cfg.max_seq_length)
    data_loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataclass_data_collator)

    cfg.datamodule.valid = {"dummy": ""}
    cfg.datamodule.test = {"dummy": ""}
    module = SenterModule(cfg)

    trainer.fit(model=module, train_dataloaders=data_loader, val_dataloaders=data_loader)
    trainer.test(model=module, dataloaders=data_loader)
    trainer.predict(model=module, dataloaders=data_loader)