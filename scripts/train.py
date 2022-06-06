import os
from typing import Optional

import hydra
import pytorch_lightning as pl
import transformers.utils.logging as hf_logging
import wandb
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LightningLoggerBase

from jula.datamodule.datamodule import DataModule
from jula.models.char_module import CharModule
from jula.models.word_module import WordModule

hf_logging.set_verbosity(hf_logging.ERROR)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    if isinstance(cfg.devices, str):
        try:
            cfg.devices = [int(x) for x in cfg.devices.split(",")]
        except ValueError:
            cfg.devices = None
    cfg.seed = pl.seed_everything(seed=cfg.seed, workers=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    is_debug: bool = True if "fast_dev_run" in cfg.trainer else False
    logger: Optional[LightningLoggerBase] = (
        hydra.utils.instantiate(cfg.logger) if not is_debug else None
    )
    callbacks: list[Callback] = list(
        map(hydra.utils.instantiate, cfg.get("callbacks", {}).values())
    )

    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
        devices=cfg.devices,
    )

    datamodule: DataModule = DataModule(cfg=cfg)

    if cfg.module.type == "char":
        model: CharModule = CharModule(hparams=cfg)
    elif cfg.module.type == "word":
        model: WordModule = WordModule(hparams=cfg)
    else:
        raise ValueError("invalid module type")

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(
        model=model, datamodule=datamodule, ckpt_path="best" if not is_debug else None
    )
    if cfg.do_predict_after_train:
        trainer.predict(
            model=model,
            datamodule=datamodule,
            ckpt_path=trainer.checkpoint_callback.best_model_path,
        )

    wandb.finish()


if __name__ == "__main__":
    main()
