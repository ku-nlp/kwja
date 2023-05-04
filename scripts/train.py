import logging
import math
import warnings
from typing import List, Union

import hydra
import pytorch_lightning as pl
import torch
import transformers.utils.logging as hf_logging
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from kwja.datamodule.datamodule import DataModule

hf_logging.set_verbosity(hf_logging.ERROR)
logging.getLogger("rhoknp").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message=(
        r"It is recommended to use .+ when logging on epoch level in distributed setting to accumulate the metric"
        r" across devices"
    ),
    category=PossibleUserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=(
        r"Using `DistributedSampler` with the dataloaders. During `trainer..+`, it is recommended to use"
        r" `Trainer(devices=1, num_nodes=1)` to ensure each sample/batch gets evaluated exactly once. Otherwise,"
        r" multi-device settings use `DistributedSampler` that replicates some samples to make sure all devices have"
        r" same batch size in case of uneven inputs."
    ),
    category=PossibleUserWarning,
)
OmegaConf.register_new_resolver("concat", lambda x, y: x + y)


@hydra.main(version_base=None, config_path="../configs")
def main(cfg: DictConfig):
    load_dotenv()
    if isinstance(cfg.devices, str):
        cfg.devices = list(map(int, cfg.devices.split(","))) if "," in cfg.devices else int(cfg.devices)
    if isinstance(cfg.max_batches_per_device, str):
        cfg.max_batches_per_device = int(cfg.max_batches_per_device)
    if isinstance(cfg.num_workers, str):
        cfg.num_workers = int(cfg.num_workers)
    cfg.seed = pl.seed_everything(seed=cfg.seed, workers=True)

    logger: Union[Logger, bool] = cfg.get("logger", False) and hydra.utils.instantiate(cfg.get("logger"))
    callbacks: List[Callback] = list(map(hydra.utils.instantiate, cfg.get("callbacks", {}).values()))

    # Calculate gradient_accumulation_steps assuming DDP
    num_devices: int = 1
    if isinstance(cfg.devices, (list, ListConfig)):
        num_devices = len(cfg.devices)
    elif isinstance(cfg.devices, int):
        num_devices = cfg.devices
    cfg.trainer.accumulate_grad_batches = math.ceil(
        cfg.effective_batch_size / (cfg.max_batches_per_device * num_devices)
    )
    batches_per_device = cfg.effective_batch_size // (num_devices * cfg.trainer.accumulate_grad_batches)
    # if effective_batch_size % (accumulate_grad_batches * num_devices) != 0, then
    # cause an error of at most accumulate_grad_batches * num_devices compared in effective_batch_size
    # otherwise, no error
    cfg.effective_batch_size = batches_per_device * num_devices * cfg.trainer.accumulate_grad_batches
    cfg.datamodule.batch_size = batches_per_device

    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
        devices=cfg.devices,
    )

    datamodule = DataModule(cfg=cfg.datamodule)

    model: pl.LightningModule = hydra.utils.instantiate(cfg.module.cls, hparams=cfg, _recursive_=False)
    if cfg.compile is True:
        model = torch.compile(model)  # type: ignore

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best" if not trainer.fast_dev_run else None)
    if cfg.do_predict_after_train:
        trainer.predict(
            model=model,
            dataloaders=datamodule.val_dataloader(),
            ckpt_path=trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else "best",
        )

    wandb.finish()


if __name__ == "__main__":
    main()
