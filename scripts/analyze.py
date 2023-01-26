import sys
from pathlib import Path
from typing import List

import hydra
import pytorch_lightning as pl
from dotenv import load_dotenv
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import TrainerFn

from kwja.cli.utils import suppress_debug_info
from kwja.datamodule.datamodule import DataModule

suppress_debug_info()
OmegaConf.register_new_resolver("concat", lambda x, y: x + y)


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(eval_cfg: DictConfig):
    load_dotenv()
    if isinstance(eval_cfg.devices, str):
        try:
            eval_cfg.devices = [int(x) for x in eval_cfg.devices.split(",")]
        except ValueError:
            eval_cfg.devices = None

    # Load saved model and config
    model: pl.LightningModule = hydra.utils.call(eval_cfg.module.load_from_checkpoint, _recursive_=False)

    train_cfg: DictConfig = model.hparams
    OmegaConf.set_struct(train_cfg, False)  # enable to add new key-value pairs
    cfg = OmegaConf.merge(train_cfg, eval_cfg)
    assert isinstance(cfg, DictConfig)

    callbacks: List[Callback] = []
    for k, v in cfg.get("callbacks", {}).items():
        if k == "prediction_writer":
            callbacks.append(hydra.utils.instantiate(v, use_stdout=True))
        elif k == "progress_bar":
            callbacks.append(hydra.utils.instantiate(v))

    num_devices: int = len(cfg.devices) if isinstance(cfg.devices, (list, ListConfig)) else cfg.devices
    cfg.effective_batch_size = cfg.max_batches_per_device * num_devices
    cfg.datamodule.batch_size = cfg.max_batches_per_device

    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=None,
        callbacks=callbacks,
        devices=cfg.devices,
    )

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
