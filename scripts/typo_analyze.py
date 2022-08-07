import sys

import hydra
import pytorch_lightning as pl
from dotenv import load_dotenv
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import TrainerFn

from jula.cli.utils import suppress_debug_info
from jula.datamodule.datamodule import DataModule
from jula.models.typo_module import TypoModule

suppress_debug_info()


@hydra.main(version_base=None, config_path="../configs", config_name="typo_module")
def main(cfg: DictConfig):
    load_dotenv()
    if isinstance(cfg.devices, str):
        try:
            cfg.devices = [int(x) for x in cfg.devices.split(",")]
        except ValueError:
            cfg.devices = None

    callbacks: list[Callback] = []
    for k, v in cfg.get("callbacks", {}).items():
        if k == "prediction_writer":
            callbacks.append(hydra.utils.instantiate(v, use_stdout=True))

    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=None,
        enable_progress_bar=False,
        callbacks=callbacks,
        devices=cfg.devices,
    )

    if cfg.config_name in cfg.module.typo:
        model = TypoModule.load_from_checkpoint(checkpoint_path=cfg.checkpoint_path, hparams=cfg)
    else:
        raise ValueError("invalid config name")

    cfg.datamodule.predict.texts = sys.stdin.readlines()
    datamodule = DataModule(cfg=cfg)
    datamodule.setup(stage=TrainerFn.PREDICTING)

    trainer.predict(model=model, dataloaders=datamodule.predict_dataloader())


if __name__ == "__main__":
    main()
