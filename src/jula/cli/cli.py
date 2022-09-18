from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import hydra
import pytorch_lightning as pl
import typer
from omegaconf import OmegaConf
from pytorch_lightning.trainer.states import TrainerFn

from jula.cli.utils import download_checkpoint_from_url, suppress_debug_info
from jula.datamodule.datamodule import DataModule
from jula.models.char_module import CharModule
from jula.models.typo_module import TypoModule
from jula.models.word_module import WordModule

TYPO_CHECKPOINT_URL = "https://lotus.kuee.kyoto-u.ac.jp/kwja/typo_roberta-base-wwm_seq512.ckpt"
CHAR_CHECKPOINT_URL = "https://lotus.kuee.kyoto-u.ac.jp/kwja/char_roberta-base-wwm_seq512.ckpt"
WORD_CHECKPOINT_URL = "https://lotus.kuee.kyoto-u.ac.jp/kwja/word_roberta-base_seq256.ckpt"

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

    tmp_dir: TemporaryDirectory = TemporaryDirectory()

    # typo
    typo_checkpoint_path: Path = download_checkpoint_from_url(TYPO_CHECKPOINT_URL)
    typo_model: TypoModule = TypoModule.load_from_checkpoint(str(typo_checkpoint_path))
    typo_cfg = typo_model.hparams
    typo_trainer: pl.Trainer = pl.Trainer(
        logger=False,
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
    char_checkpoint_path: Path = download_checkpoint_from_url(CHAR_CHECKPOINT_URL)
    char_model: CharModule = CharModule.load_from_checkpoint(str(char_checkpoint_path))
    char_cfg = char_model.hparams
    char_trainer: pl.Trainer = pl.Trainer(
        logger=False,
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
    char_cfg.datamodule.predict.texts = Path(f"{tmp_dir.name}/predict_typo.txt").read_text().splitlines()
    char_datamodule = DataModule(cfg=char_cfg.datamodule)
    char_datamodule.setup(stage=TrainerFn.PREDICTING)
    char_trainer.predict(model=char_model, dataloaders=char_datamodule.predict_dataloader())
    del char_model

    # word
    word_checkpoint_path: Path = download_checkpoint_from_url(WORD_CHECKPOINT_URL)
    word_model: WordModule = WordModule.load_from_checkpoint(str(word_checkpoint_path))
    word_cfg = word_model.hparams
    word_trainer: pl.Trainer = pl.Trainer(
        logger=False,
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
