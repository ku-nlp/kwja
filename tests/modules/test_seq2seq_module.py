from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from kwja.datamodule.datamodule import dataclass_data_collator
from kwja.datamodule.datasets import Seq2SeqDataset
from kwja.modules import Seq2SeqModule

with initialize(version_base=None, config_path="../../configs"):
    cfg = compose(
        config_name="seq2seq_module.debug",
        return_hydra_config=True,
        overrides=["max_src_length=32", "max_tgt_length=512"],
    )
    HydraConfig.instance().set_config(cfg)
    OmegaConf.set_readonly(cfg.hydra, False)
    OmegaConf.resolve(cfg)


def test_init() -> None:
    _ = Seq2SeqModule(cfg)


def test_steps(fixture_data_dir: Path) -> None:
    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=False,
        enable_checkpointing=False,
        devices=1,
        accelerator="cpu",
    )

    path = fixture_data_dir / "datasets" / "word_files"
    seq2seq_tokenizer = hydra.utils.instantiate(cfg.datamodule.train.kyoto.tokenizer)
    dataset = Seq2SeqDataset(str(path), seq2seq_tokenizer, cfg.max_src_length, cfg.max_tgt_length)
    data_loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataclass_data_collator)
    val_dataloaders = {"dummy": data_loader}

    cfg.datamodule.valid = {"dummy": ""}
    cfg.datamodule.test = {"dummy": ""}
    module = Seq2SeqModule(cfg)

    trainer.fit(model=module, train_dataloaders=data_loader, val_dataloaders=val_dataloaders)
    trainer.test(model=module, dataloaders=val_dataloaders)
    trainer.predict(model=module, dataloaders=val_dataloaders)
