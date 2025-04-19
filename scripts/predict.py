import os

import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import DictConfig, ListConfig, OmegaConf

from kwja.datamodule.datamodule import DataModule
from kwja.utils.logging_util import filter_logs

filter_logs(environment="development")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(eval_cfg: DictConfig) -> None:
    load_dotenv()
    if isinstance(eval_cfg.devices, str):
        eval_cfg.devices = (
            list(map(int, eval_cfg.devices.split(","))) if "," in eval_cfg.devices else int(eval_cfg.devices)
        )
    if isinstance(eval_cfg.max_batches_per_device, str):
        eval_cfg.max_batches_per_device = int(eval_cfg.max_batches_per_device)
    if isinstance(eval_cfg.num_workers, str):
        eval_cfg.num_workers = int(eval_cfg.num_workers)

    # Load saved model and config
    model: L.LightningModule = hydra.utils.call(eval_cfg.module.load_from_checkpoint, _recursive_=False)
    if eval_cfg.compile is True:
        model = torch.compile(model)  # type: ignore

    train_cfg: DictConfig = model.hparams
    OmegaConf.set_struct(train_cfg, False)  # enable to add new key-value pairs
    cfg = OmegaConf.merge(train_cfg, eval_cfg)
    assert isinstance(cfg, DictConfig)

    callbacks: list[Callback] = []
    for k, v in cfg.get("callbacks", {}).items():
        if k in {"prediction_writer", "progress_bar"}:
            callbacks.append(hydra.utils.instantiate(v))

    num_devices: int = len(cfg.devices) if isinstance(cfg.devices, (list, ListConfig)) else cfg.devices
    cfg.effective_batch_size = cfg.max_batches_per_device * num_devices
    cfg.datamodule.batch_size = cfg.max_batches_per_device

    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=False,
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
    trainer.predict(model=model, dataloaders=dataloader)


if __name__ == "__main__":
    main()
