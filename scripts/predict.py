import hydra
import pytorch_lightning as pl
import transformers.utils.logging as hf_logging
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import TrainerFn

from kwja.datamodule.datamodule import DataModule

hf_logging.set_verbosity(hf_logging.ERROR)
OmegaConf.register_new_resolver("concat", lambda x, y: x + y)


@hydra.main(version_base=None, config_path="../configs")
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
            callbacks.append(hydra.utils.instantiate(v))

    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=None,
        callbacks=callbacks,
        devices=cfg.devices,
    )
    model: pl.LightningModule = hydra.utils.call(cfg.module.load_from_checkpoint, hparams=cfg, _recursive_=False)

    datamodule = DataModule(cfg=cfg.datamodule)
    datamodule.setup(stage=TrainerFn.TESTING)
    dataloader = datamodule.test_dataloader()  # or val_dataloader()
    trainer.predict(model=model, dataloaders=dataloader)


if __name__ == "__main__":
    main()
