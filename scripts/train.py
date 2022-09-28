import logging
import math
from typing import Optional

import hydra
import pytorch_lightning as pl
import transformers.utils.logging as hf_logging
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LightningLoggerBase

from kwja.datamodule.datamodule import DataModule

hf_logging.set_verbosity(hf_logging.ERROR)
logging.getLogger("rhoknp").setLevel(logging.ERROR)
OmegaConf.register_new_resolver("concat", lambda x, y: x + y)


@hydra.main(version_base=None, config_path="../configs")
def main(cfg: DictConfig):
    load_dotenv()
    if isinstance(cfg.devices, str):
        try:
            cfg.devices = [int(x) for x in cfg.devices.split(",")]
        except ValueError:
            cfg.devices = None
    cfg.seed = pl.seed_everything(seed=cfg.seed, workers=True)

    is_debug: bool = True if "fast_dev_run" in cfg.trainer else False
    logger: Optional[LightningLoggerBase] = hydra.utils.instantiate(cfg.logger) if not is_debug else None
    callbacks: list[Callback] = list(map(hydra.utils.instantiate, cfg.get("callbacks", {}).values()))

    # Calculate gradient_accumulation_steps assuming DDP
    num_devices: int = len(cfg.devices) if isinstance(cfg.devices, (list, ListConfig)) else cfg.devices
    cfg.trainer.accumulate_grad_batches = math.ceil(
        cfg.effective_batch_size / (cfg.max_batches_per_device * num_devices)
    )
    batches_per_device = cfg.effective_batch_size // (num_devices * cfg.trainer.accumulate_grad_batches)
    # if effective_batch_size % (accumulate_grad_batches * num_devices) != 0, then
    # error of at most accumulate_grad_batches * num_devices compared to the original effective_batch_size
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

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best" if not is_debug else None)
    if cfg.do_predict_after_train:
        trainer.predict(
            model=model,
            dataloaders=datamodule.val_dataloader(),
            ckpt_path=trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else "best",
        )

    wandb.finish()


if __name__ == "__main__":
    main()
