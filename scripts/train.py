import logging
from typing import Optional, Union

import hydra
import pytorch_lightning as pl
import transformers.utils.logging as hf_logging
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LightningLoggerBase

from jula.datamodule.datamodule import DataModule
from jula.models.char_module import CharModule
from jula.models.typo_module import TypoModule
from jula.models.word_module import WordModule

hf_logging.set_verbosity(hf_logging.ERROR)
logging.getLogger("rhoknp").setLevel(logging.WARNING)
OmegaConf.register_new_resolver("concat", lambda x, y: x + y)


@hydra.main(version_base=None, config_path="../configs")
def main(cfg: DictConfig):
    load_dotenv()
    if isinstance(cfg.devices, str):
        try:
            cfg.devices = [int(x) for x in cfg.devices.split(",")]
        except ValueError:
            cfg.devices = None
    cfg.seed = pl.seed_everything(seed=cfg.seed, workers=True)

    is_debug: bool = True if "fast_dev_run" in cfg.trainer else False
    logger: Optional[LightningLoggerBase] = hydra.utils.instantiate(cfg.logger) if not is_debug else None
    callbacks: list[Callback] = list(map(hydra.utils.instantiate, cfg.get("callbacks", {}).values()))

    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
        devices=cfg.devices,
    )

    datamodule = DataModule(cfg=cfg.datamodule)

    model: Union[TypoModule, CharModule, WordModule]
    if cfg.config_name in cfg.module.typo:
        model = TypoModule(hparams=cfg)
    elif cfg.config_name in cfg.module.char:
        model = CharModule(hparams=cfg)
    elif cfg.config_name in cfg.module.word:
        model = WordModule(hparams=cfg)
    else:
        raise ValueError(f"invalid config name: `{cfg.config_name}`")

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best" if not is_debug else None)
    if cfg.do_predict_after_train:
        trainer.predict(
            model=model,
            datamodule=datamodule,
            ckpt_path=trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else "best",
        )

    wandb.finish()


if __name__ == "__main__":
    main()
