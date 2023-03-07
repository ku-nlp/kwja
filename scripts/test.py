import logging
import warnings
from typing import List, Optional

import hydra
import pytorch_lightning as pl
import transformers.utils.logging as hf_logging
from dotenv import load_dotenv
from lightning_fabric.utilities.warnings import PossibleUserWarning
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.trainer.states import TrainerFn

from kwja.datamodule.datamodule import DataModule

hf_logging.set_verbosity(hf_logging.ERROR)
logging.getLogger("rhoknp").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    r"It is recommended to use .+ when logging on epoch level in distributed setting to accumulate the metric across"
    r" devices",
    category=PossibleUserWarning,
)
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

    logger: Optional[Logger] = cfg.get("logger") and hydra.utils.instantiate(cfg.get("logger"))
    callbacks: List[Callback] = list(map(hydra.utils.instantiate, cfg.get("callbacks", {}).values()))

    num_devices: int = len(cfg.devices) if isinstance(cfg.devices, (list, ListConfig)) else cfg.devices
    cfg.effective_batch_size = cfg.max_batches_per_device * num_devices
    cfg.datamodule.batch_size = cfg.max_batches_per_device

    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
        devices=cfg.devices,
    )

    datamodule = DataModule(cfg=cfg.datamodule)
    datamodule.setup(stage=TrainerFn.TESTING)
    if cfg.eval_set == "test":
        dataloader = datamodule.test_dataloader()
    elif cfg.eval_set == "valid":
        dataloader = datamodule.val_dataloader()
    else:
        raise ValueError(f"invalid eval_set: {cfg.eval_set}")
    trainer.test(model=model, dataloaders=dataloader)


if __name__ == "__main__":
    main()
