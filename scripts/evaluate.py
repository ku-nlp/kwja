import os
from pathlib import Path
from typing import Union

import hydra
import pytorch_lightning as pl
import transformers.utils.logging as hf_logging
from omegaconf import DictConfig

from jula.datamodule.datamodule import DataModule
from jula.models.char_module import CharModule
from jula.models.word_module import WordModule

hf_logging.set_verbosity(hf_logging.ERROR)


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg: DictConfig):
    if isinstance(cfg.devices, str):
        try:
            cfg.devices = [int(x) for x in cfg.devices.split(",")]
        except ValueError:
            cfg.devices = None
    cfg.seed = int(Path(cfg.checkpoint_path).parent.stem)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=None,
        callbacks=hydra.utils.instantiate(cfg.callbacks.prediction_writer),
        devices=cfg.devices,
    )

    model: Union[CharModule, WordModule]
    if cfg.module.type in ["char", "char_typo"]:
        model = CharModule.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_path, hparams=cfg
        )
    elif cfg.module.type == "word":
        model = WordModule.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_path, hparams=cfg
        )
    else:
        raise ValueError("invalid module type")

    datamodule: DataModule = DataModule(cfg=cfg)
    trainer.predict(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
