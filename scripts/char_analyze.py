import hydra
import pytorch_lightning as pl
import transformers.utils.logging as hf_logging
from dotenv import load_dotenv
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader

from jula.datamodule.datasets.raw_text_dataset import RawTextDataset
from jula.models.char_module import CharModule

hf_logging.set_verbosity(hf_logging.ERROR)


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
        callbacks=callbacks,
        devices=cfg.devices,
    )

    if cfg.config_name in cfg.module.char:
        model = CharModule.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_path, hparams=cfg
        )
    else:
        raise ValueError("invalid config name")

    inp = input()

    # TODO: Use hydra for configuration
    dataset = RawTextDataset(
        inp.split("\n"),
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
