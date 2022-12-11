import sys
from pathlib import Path
from typing import List

import hydra
import pytorch_lightning as pl
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import TrainerFn

from kwja.cli.utils import suppress_debug_info
from kwja.datamodule.datamodule import DataModule

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

    callbacks: List[Callback] = [hydra.utils.instantiate(cfg.callbacks.prediction_writer, use_stdout=True)]

    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=None,
        enable_progress_bar=False,
        callbacks=callbacks,
        devices=cfg.devices,
    )

    model: pl.LightningModule = hydra.utils.call(cfg.module.load_from_checkpoint, hparams=cfg, _recursive_=False)

    if getattr(cfg.datamodule.predict, "knp_file", None):
        cfg.datamodule.predict.knp_file = Path(cfg.datamodule.predict.knp_file)
    elif getattr(cfg.datamodule.predict, "juman_file", None):
        cfg.datamodule.predict.juman_file = Path(cfg.datamodule.predict.juman_file)
    elif not cfg.datamodule.predict.texts:
        cfg.datamodule.predict.texts = sys.stdin.read().splitlines()
    datamodule = DataModule(cfg=cfg.datamodule)
    datamodule.setup(stage=TrainerFn.PREDICTING)

    if getattr(cfg, "load_only", False):
        sys.exit(0)
    trainer.predict(model=model, dataloaders=datamodule.predict_dataloader(), return_predictions=False)


if __name__ == "__main__":
    main()
