import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import hydra
import pytorch_lightning as pl
import typer
from dotenv import load_dotenv
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.trainer.states import TrainerFn

from jula.cli.utils import suppress_debug_info
from jula.datamodule.datamodule import DataModule

suppress_debug_info()
OmegaConf.register_new_resolver("concat", lambda x, y: x + y)

app = typer.Typer()


@app.command()
def main(
    text: Optional[str] = typer.Option(None, help="Text to be analyzed."),
    filename: Optional[Path] = typer.Option(None, help="File to be analyzed."),
) -> None:
    if text is not None and filename is not None:
        typer.echo("ERROR: Please provide text or filename, not both")
        raise typer.Exit()
    elif text is not None:
        input_texts: list[str] = text.splitlines()
    elif filename is not None:
        with Path(filename).open() as f:
            input_texts = [line.strip() for line in f]
    else:
        typer.echo("ERROR: Please provide text or filename")
        raise typer.Exit

    load_dotenv()
    env_model_dir: Optional[str] = os.getenv("MODEL_DIR")
    if env_model_dir is None:
        typer.echo("ERROR: Please set the MODEL_DIR environment variable")
        raise typer.Exit()
    model_dir: Path = Path(env_model_dir)

    tmp_dir: TemporaryDirectory = TemporaryDirectory()

    # typo
    with initialize(version_base=None, config_path="../../../configs"):
        typo_cfg: DictConfig = compose(config_name="typo_module")
    typo_trainer: pl.Trainer = pl.Trainer(
        logger=None,
        enable_progress_bar=False,
        callbacks=[
            hydra.utils.instantiate(
                typo_cfg.callbacks.prediction_writer,
                output_dir=str(tmp_dir.name),
                pred_filename="predict_typo",
            )
        ],
        devices=1,
    )
    typo_cfg.module.load_from_checkpoint.checkpoint_path = str((model_dir / "typo.ckpt").resolve())
    typo_model: pl.LightningModule = hydra.utils.call(typo_cfg.module.load_from_checkpoint)

    typo_cfg.datamodule.predict.texts = input_texts
    typo_datamodule = DataModule(cfg=typo_cfg.datamodule)
    typo_datamodule.setup(stage=TrainerFn.PREDICTING)
    typo_trainer.predict(model=typo_model, dataloaders=typo_datamodule.predict_dataloader())
    del typo_model

    # char
    with initialize(version_base=None, config_path="../../../configs"):
        char_cfg: DictConfig = compose(config_name="char_module")
    char_trainer: pl.Trainer = pl.Trainer(
        logger=None,
        enable_progress_bar=False,
        callbacks=[
            hydra.utils.instantiate(
                char_cfg.callbacks.prediction_writer,
                output_dir=str(tmp_dir.name),
                pred_filename="predict_char",
            )
        ],
        devices=1,
    )
    char_cfg.module.load_from_checkpoint.checkpoint_path = str((model_dir / "char.ckpt").resolve())
    char_model: pl.LightningModule = hydra.utils.call(char_cfg.module.load_from_checkpoint)
    with open(f"{tmp_dir.name}/predict_typo.txt") as f:
        typo_results = [line.strip() for line in f]
    char_cfg.datamodule.predict.texts = typo_results
    char_datamodule = DataModule(cfg=char_cfg.datamodule)
    char_datamodule.setup(stage=TrainerFn.PREDICTING)
    char_trainer.predict(model=char_model, dataloaders=char_datamodule.predict_dataloader())
    del char_model

    # word
    with initialize(version_base=None, config_path="../../../configs"):
        word_cfg: DictConfig = compose(config_name="word_module")
    word_trainer: pl.Trainer = pl.Trainer(
        logger=None,
        enable_progress_bar=False,
        callbacks=[
            hydra.utils.instantiate(
                word_cfg.callbacks.prediction_writer,
                output_dir=str(tmp_dir.name),
                pred_filename="predict_word",
                use_stdout=True,
            )
        ],
        devices=1,
    )
    word_cfg.module.load_from_checkpoint.checkpoint_path = str((model_dir / "word.ckpt").resolve())
    word_model: pl.LightningModule = hydra.utils.call(word_cfg.module.load_from_checkpoint)

    with open(f"{tmp_dir.name}/predict_char.txt") as f:
        char_results = [line.strip() for line in f]
    word_cfg.datamodule.predict.texts = [x for i, x in enumerate(char_results) if i % 2 == 1]
    word_datamodule = DataModule(cfg=word_cfg.datamodule)
    word_datamodule.setup(stage=TrainerFn.PREDICTING)
    word_trainer.predict(model=word_model, dataloaders=word_datamodule.predict_dataloader())
    tmp_dir.cleanup()
