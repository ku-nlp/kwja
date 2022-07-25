import sys

import hydra
import pytorch_lightning as pl
from dotenv import load_dotenv
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader

from jula.cli.utils import suppress_debug_info
from jula.datamodule.datasets.raw_text_dataset import RawTextDataset
from jula.models.char_module import CharModule

suppress_debug_info()


@hydra.main(version_base=None, config_path="../configs", config_name="word_segmenter")
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
            callbacks.append(hydra.utils.instantiate(v, use_stdout=True))

    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=None,
        enable_progress_bar=False,
        callbacks=callbacks,
        devices=cfg.devices,
    )

    if cfg.config_name in cfg.module.char:
        model = CharModule.load_from_checkpoint(checkpoint_path=cfg.checkpoint_path, hparams=cfg)
    else:
        raise ValueError("invalid config name")

    # TODO: Use hydra for configuration
    dataset = RawTextDataset(
        sys.stdin.readlines(),
        model_name_or_path=cfg.datamodule.model_name_or_path,
        max_seq_length=cfg.datamodule.max_seq_length,
        tokenizer_kwargs=cfg.datamodule.tokenizer_kwargs,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    trainer.predict(model=model, dataloaders=dataloader)


if __name__ == "__main__":
    main()
