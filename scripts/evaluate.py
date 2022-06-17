from pathlib import Path
from typing import Union

import hydra
import pytorch_lightning as pl
import transformers.utils.logging as hf_logging
from dotenv import load_dotenv
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback

from jula.datamodule.datamodule import DataModule
from jula.models.char_module import CharModule
from jula.models.typo_module import TypoModule
from jula.models.word_module import WordModule

hf_logging.set_verbosity(hf_logging.ERROR)


@hydra.main(version_base=None, config_path="../configs", config_name="word_segmentor")
def main(cfg: DictConfig):
    load_dotenv()
    if isinstance(cfg.devices, str):
        try:
            cfg.devices = [int(x) for x in cfg.devices.split(",")]
        except ValueError:
            cfg.devices = None
    cfg.seed = int(Path(cfg.checkpoint_path).parent.stem)

    callbacks: list[Callback] = []
    for k, v in cfg.get("callbacks", {}).items():
        if k == "prediction_writer":
            callbacks.append(hydra.utils.instantiate(v))

    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=None,
        callbacks=callbacks,
        devices=cfg.devices,
    )

    model: Union[TypoModule, CharModule, WordModule]
    # add config name if you want
    if cfg.config_name in cfg.module.config_names:
        model = TypoModule.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_path, hparams=cfg
        )
    elif cfg.config_name in cfg.module.config_names:
        model = CharModule.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_path, hparams=cfg
        )
    elif cfg.config_name in cfg.module.config_names:
        model = WordModule.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_path, hparams=cfg
        )
    else:
        raise ValueError("invalid config name")

    datamodule: DataModule = DataModule(cfg=cfg)
    trainer.predict(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
