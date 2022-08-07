import sys
from typing import Union

import hydra
import pytorch_lightning as pl
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import TrainerFn

from jula.cli.utils import suppress_debug_info
from jula.datamodule.datamodule import DataModule
from jula.models.char_module import CharModule
from jula.models.typo_module import TypoModule
from jula.models.word_module import WordModule

suppress_debug_info()
OmegaConf.register_new_resolver("concat", lambda x, y: x + y)


@hydra.main(version_base=None, config_path="../configs")
def main(cfg: DictConfig):
    load_dotenv()
    if isinstance(cfg.devices, str):
        try:
            cfg.devices = [int(x) for x in cfg.devices.split(",")]
        except ValueError:
            cfg.devices = None

    callbacks: list[Callback] = [hydra.utils.instantiate(cfg.callbacks.prediction_writer, use_stdout=True)]

    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=None,
        enable_progress_bar=False,
        callbacks=callbacks,
        devices=cfg.devices,
    )

    model: Union[TypoModule, CharModule, WordModule]
    if cfg.config_name in cfg.module.typo:
        model = TypoModule.load_from_checkpoint(checkpoint_path=cfg.checkpoint_path, hparams=cfg)
    elif cfg.config_name in cfg.module.char:
        model = CharModule.load_from_checkpoint(checkpoint_path=cfg.checkpoint_path, hparams=cfg)
    elif cfg.config_name in cfg.module.word:
        model = WordModule.load_from_checkpoint(checkpoint_path=cfg.checkpoint_path, hparams=cfg)
    else:
        raise ValueError(f"invalid config name: `{cfg.config_name}`")

    cfg.datamodule.predict.texts = sys.stdin.readlines()
    datamodule = DataModule(cfg=cfg.datamodule)
    datamodule.setup(stage=TrainerFn.PREDICTING)

    trainer.predict(model=model, dataloaders=datamodule.predict_dataloader())


if __name__ == "__main__":
    main()
