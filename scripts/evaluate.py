import os

import hydra
import pytorch_lightning as pl
import transformers.utils.logging as hf_logging
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback

from jula.datamodule.datamodule import DataModule
from jula.models.char_module import CharModule
from jula.models.word_module import WordModule

hf_logging.set_verbosity(hf_logging.ERROR)


@hydra.main(version_base=None, config_path="../configs", config_name="predict")
def main(cfg: DictConfig):
    if isinstance(cfg.devices, str):
        try:
            cfg.devices = [int(x) for x in cfg.devices.split(",")]
        except ValueError:
            cfg.devices = None
    cfg.seed = pl.seed_everything(seed=cfg.seed, workers=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    callbacks: list[Callback] = list(
        map(hydra.utils.instantiate, cfg.get("callbacks", {}).values())
    )

    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=None,
        callbacks=callbacks,
        devices=cfg.devices,
    )

    if cfg.module.type == "char":
        model: CharModule = CharModule.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_path, hparams=cfg
        )
    elif cfg.module.type == "word":
        model: WordModule = WordModule.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_path, hparams=cfg
        )
    else:
        raise ValueError("invalid module type")

    datamodule: DataModule = DataModule(cfg=cfg, global_rank=trainer.global_rank)
    trainer.predict(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
