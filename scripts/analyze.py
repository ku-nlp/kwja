import sys

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

    callbacks: list[Callback] = [hydra.utils.instantiate(cfg.callbacks.prediction_writer, use_stdout=True)]

    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=None,
        enable_progress_bar=False,
        callbacks=callbacks,
        devices=cfg.devices,
    )

    model: pl.LightningModule = hydra.utils.call(cfg.module.load_from_checkpoint, hparams=cfg, _recursive_=False)

    if not cfg.datamodule.predict.texts:
        cfg.datamodule.predict.texts = sys.stdin.read().splitlines()
    datamodule = DataModule(cfg=cfg.datamodule)
    datamodule.setup(stage=TrainerFn.PREDICTING)

    trainer.predict(model=model, dataloaders=datamodule.predict_dataloader())


if __name__ == "__main__":
    main()
