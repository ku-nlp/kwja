import logging
import math
from typing import List, Optional

import hydra
import pytorch_lightning as pl
import transformers.utils.logging as hf_logging
from dotenv import load_dotenv
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.trainer.states import TrainerFn

from kwja.datamodule.datamodule import DataModule

hf_logging.set_verbosity(hf_logging.ERROR)
logging.getLogger("rhoknp").setLevel(logging.ERROR)
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

    is_debug: bool = True if "fast_dev_run" in cfg.trainer else False
    logger: Optional[LightningLoggerBase] = hydra.utils.instantiate(cfg.logger) if is_debug is False else None
    callbacks: List[Callback] = list(map(hydra.utils.instantiate, cfg.get("callbacks", {}).values()))

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
    datamodule.setup(stage=TrainerFn.TESTING)
    dataloader = datamodule.test_dataloader()  # or val_dataloader()
    trainer.test(model=model, dataloaders=dataloader)


if __name__ == "__main__":
    main()
