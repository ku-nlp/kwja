import os
import sys
from pathlib import Path
from time import time
from typing import List

import hydra
import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import TrainerFn

from kwja.cli.cli import normalize_text
from kwja.datamodule.datamodule import DataModule
from kwja.utils.logging_util import filter_logs

filter_logs(environment="production")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(eval_cfg: DictConfig):
    load_dotenv()
    if isinstance(eval_cfg.devices, str):
        eval_cfg.devices = (
            list(map(int, eval_cfg.devices.split(","))) if "," in eval_cfg.devices else int(eval_cfg.devices)
        )

    # Load saved model and config
    model: pl.LightningModule = hydra.utils.call(eval_cfg.module.load_from_checkpoint, _recursive_=False)
    if eval_cfg.compile is True:
        model = torch.compile(model)  # type: ignore

    start = time()

    train_cfg: DictConfig = model.hparams
    OmegaConf.set_struct(train_cfg, False)  # enable to add new key-value pairs
    cfg = OmegaConf.merge(train_cfg, eval_cfg)
    assert isinstance(cfg, DictConfig)

    callbacks: List[Callback] = []
    for k, v in cfg.get("callbacks", {}).items():
        if k in {"prediction_writer", "progress_bar"}:
            callbacks.append(hydra.utils.instantiate(v))

    num_devices: int = len(cfg.devices) if isinstance(cfg.devices, (list, ListConfig)) else cfg.devices
    cfg.effective_batch_size = cfg.max_batches_per_device * num_devices
    cfg.datamodule.batch_size = cfg.max_batches_per_device

    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=False,
        callbacks=callbacks,
        devices=cfg.devices,
    )

    if getattr(cfg.datamodule.predict, "knp_file", None):
        cfg.datamodule.predict.knp_file = Path(cfg.datamodule.predict.knp_file)
    elif getattr(cfg.datamodule.predict, "juman_file", None):
        cfg.datamodule.predict.juman_file = Path(cfg.datamodule.predict.juman_file)
    elif getattr(cfg.datamodule.predict, "texts", None) is not None:
        cfg.datamodule.predict.texts = [normalize_text(line) for line in sys.stdin.readlines()]
    datamodule = DataModule(cfg=cfg.datamodule)
    datamodule.setup(stage=TrainerFn.PREDICTING)

    trainer.predict(model=model, dataloaders=[datamodule.predict_dataloader()], return_predictions=False)

    print(f"real {time() - start:.02f}", file=sys.stderr)


if __name__ == "__main__":
    main()
