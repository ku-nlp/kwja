import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import hydra
import pytorch_lightning as pl
import typer
from dotenv import load_dotenv
from omegaconf import OmegaConf
from pytorch_lightning.trainer.states import TrainerFn

from jula.cli.utils import suppress_debug_info
from jula.datamodule.datamodule import DataModule
from jula.models.char_module import CharModule
from jula.models.typo_module import TypoModule
from jula.models.word_module import WordModule

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
    typo_model: TypoModule = TypoModule.load_from_checkpoint(str((model_dir / "typo.ckpt").resolve()))
    typo_cfg = typo_model.hparams
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
    typo_cfg.datamodule.predict.texts = input_texts
    typo_datamodule = DataModule(cfg=typo_cfg.datamodule)
    typo_datamodule.setup(stage=TrainerFn.PREDICTING)
    typo_trainer.predict(model=typo_model, dataloaders=typo_datamodule.predict_dataloader())
    del typo_model

    # char
    char_model: CharModule = CharModule.load_from_checkpoint(str((model_dir / "char.ckpt").resolve()))
    char_cfg = char_model.hparams
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
    with open(f"{tmp_dir.name}/predict_typo.txt") as f:
        typo_results = [line.strip() for line in f]
    char_cfg.datamodule.predict.texts = typo_results
    char_datamodule = DataModule(cfg=char_cfg.datamodule)
    char_datamodule.setup(stage=TrainerFn.PREDICTING)
    char_trainer.predict(model=char_model, dataloaders=char_datamodule.predict_dataloader())
    del char_model

    # word
    word_model: WordModule = WordModule.load_from_checkpoint(str((model_dir / "word.ckpt").resolve()))
    word_cfg = word_model.hparams
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
    with open(f"{tmp_dir.name}/predict_char.txt") as f:
        char_results = [line.strip() for line in f]
    word_cfg.datamodule.predict.texts = [x for i, x in enumerate(char_results) if i % 2 == 1]
    word_datamodule = DataModule(cfg=word_cfg.datamodule)
    word_datamodule.setup(stage=TrainerFn.PREDICTING)
    word_trainer.predict(model=word_model, dataloaders=word_datamodule.predict_dataloader())
    tmp_dir.cleanup()
