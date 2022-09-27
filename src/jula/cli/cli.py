from importlib import resources
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import hydra
import pytorch_lightning as pl
import typer
from omegaconf import OmegaConf
from pytorch_lightning.trainer.states import TrainerFn
from rhoknp import Document

from jula.cli.utils import download_checkpoint_from_url, suppress_debug_info
from jula.datamodule.datamodule import DataModule
from jula.models.char_module import CharModule
from jula.models.typo_module import TypoModule
from jula.models.word_module import WordModule

_CHECKPOINT_BASE_URL = "https://lotus.kuee.kyoto-u.ac.jp/kwja"
TYPO_CHECKPOINT_URL = f"{_CHECKPOINT_BASE_URL}/v1.0/typo_roberta-base-wwm_seq512.ckpt"
CHAR_CHECKPOINT_URL = f"{_CHECKPOINT_BASE_URL}/v1.0/char_roberta-base-wwm_seq512.ckpt"
WORD_CHECKPOINT_URL = f"{_CHECKPOINT_BASE_URL}/v1.0/word_roberta-base_seq256.ckpt"
WORD_DISCOURSE_CHECKPOINT_URL = f"{_CHECKPOINT_BASE_URL}/v1.0/word_discourse_roberta-base_seq256.ckpt"

suppress_debug_info()
OmegaConf.register_new_resolver("concat", lambda x, y: x + y)

app = typer.Typer()


@app.command()
def main(
    text: Optional[str] = typer.Option(None, help="Text to be analyzed."),
    filename: Optional[Path] = typer.Option(None, help="File to be analyzed."),
    discourse: Optional[bool] = typer.Option(
        False, help="Whether to use a single model for discourse relation analysis"
    ),
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
    typo_path: Path = tmp_dir.name / Path("predict_typo.txt")
    char_path: Path = tmp_dir.name / Path("predict_char.txt")
    word_path: Path = tmp_dir.name / Path("predict_word.knp")
    word_discourse_path: Path = tmp_dir.name / Path("predict_word_discourse.knp")

    # typo module
    typo_checkpoint_path: Path = download_checkpoint_from_url(TYPO_CHECKPOINT_URL)
    typo_model: TypoModule = TypoModule.load_from_checkpoint(str(typo_checkpoint_path))
    typo_cfg = typo_model.hparams
    typo_cfg.callbacks.prediction_writer.extended_vocab_path = (
        resources.files("jula") / "resource/typo_correction/multi_char_vocab.txt"
    )
    typo_trainer: pl.Trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=False,
        callbacks=[
            hydra.utils.instantiate(
                typo_cfg.callbacks.prediction_writer,
                output_dir=str(tmp_dir.name),
                pred_filename=typo_path.stem,
            )
        ],
        devices=1,
    )
    typo_cfg.datamodule.predict.texts = input_texts
    typo_datamodule = DataModule(cfg=typo_cfg.datamodule)
    typo_datamodule.setup(stage=TrainerFn.PREDICTING)
    typo_trainer.predict(model=typo_model, dataloaders=typo_datamodule.predict_dataloader())
    del typo_model

    # char module
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
                pred_filename=char_path.stem,
            )
        ],
        devices=1,
    )
    char_cfg.datamodule.predict.texts = typo_path.read_text().splitlines()
    char_datamodule = DataModule(cfg=char_cfg.datamodule)
    char_datamodule.setup(stage=TrainerFn.PREDICTING)
    char_trainer.predict(model=char_model, dataloaders=char_datamodule.predict_dataloader())
    del char_model

    # word module
    word_checkpoint_path: Path = download_checkpoint_from_url(WORD_CHECKPOINT_URL)
    word_model: WordModule = WordModule.load_from_checkpoint(str(word_checkpoint_path))
    word_cfg = word_model.hparams
    word_cfg.callbacks.prediction_writer.reading_resource_path = resources.files("jula") / "resource/reading_prediction"
    word_cfg.callbacks.prediction_writer.jumandic_path = resources.files("jula") / "resource/jumandic"
    word_trainer: pl.Trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=False,
        callbacks=[
            hydra.utils.instantiate(
                word_cfg.callbacks.prediction_writer,
                output_dir=str(tmp_dir.name),
                pred_filename=word_path.stem,
            )
        ],
        devices=1,
    )
    char_results: list[str] = char_path.read_text().splitlines()
    word_cfg.datamodule.predict.texts = [x for i, x in enumerate(char_results) if i % 2 == 1]
    word_datamodule = DataModule(cfg=word_cfg.datamodule)
    word_datamodule.setup(stage=TrainerFn.PREDICTING)
    word_trainer.predict(model=word_model, dataloaders=word_datamodule.predict_dataloader())
    del word_model
    document: Document = Document.from_knp(word_path.read_text())
    if not discourse:
        print(document.to_knp())
    else:
        # word module (discourse)
        word_discourse_checkpoint_path: Path = download_checkpoint_from_url(WORD_DISCOURSE_CHECKPOINT_URL)
        word_discourse_model: WordModule = WordModule.load_from_checkpoint(str(word_discourse_checkpoint_path))
        word_discourse_cfg = word_discourse_model.hparams
        word_discourse_trainer: pl.Trainer = pl.Trainer(
            logger=None,
            enable_progress_bar=False,
            callbacks=[
                hydra.utils.instantiate(
                    word_discourse_cfg.callbacks.prediction_writer,
                    output_dir=str(tmp_dir.name),
                    pred_filename=word_discourse_path.stem,
                )
            ],
            devices=1,
        )
        word_discourse_cfg.datamodule.predict.texts = [x for i, x in enumerate(char_results) if i % 2 == 1]
        word_discourse_datamodule = DataModule(cfg=word_discourse_cfg.datamodule)
        word_discourse_datamodule.setup(stage=TrainerFn.PREDICTING)
        word_discourse_trainer.predict(
            model=word_discourse_model,
            dataloaders=word_discourse_datamodule.predict_dataloader(),
        )
        discourse_document: Document = Document.from_knp(word_discourse_path.read_text())
        for base_phrase in document.base_phrases:
            base_phrase.discourse_relation_tag = discourse_document.base_phrases[
                base_phrase.index
            ].discourse_relation_tag
        print(document.to_knp())
    tmp_dir.cleanup()
