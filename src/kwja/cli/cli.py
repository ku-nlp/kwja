from importlib import resources
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
import typer
from omegaconf import OmegaConf
from pytorch_lightning.trainer.states import TrainerFn
from rhoknp import Document

from kwja.callbacks.word_module_discourse_writer import WordModuleDiscourseWriter
from kwja.callbacks.word_module_writer import WordModuleWriter
from kwja.cli.utils import download_checkpoint_from_url, suppress_debug_info
from kwja.datamodule.datamodule import DataModule
from kwja.models.char_module import CharModule
from kwja.models.typo_module import TypoModule
from kwja.models.word_module import WordModule

_CHECKPOINT_BASE_URL = "https://lotus.kuee.kyoto-u.ac.jp/kwja"
TYPO_CHECKPOINT_URL = f"{_CHECKPOINT_BASE_URL}/v1.0/typo_roberta-base-wwm_seq512.ckpt"
CHAR_CHECKPOINT_URL = f"{_CHECKPOINT_BASE_URL}/v1.0/char_roberta-base-wwm_seq512.ckpt"
WORD_CHECKPOINT_URL = f"{_CHECKPOINT_BASE_URL}/v1.0/word_roberta-base_seq128.ckpt"
WORD_DISCOURSE_CHECKPOINT_URL = f"{_CHECKPOINT_BASE_URL}/v1.0/disc_roberta-base_seq128.ckpt"

suppress_debug_info()
OmegaConf.register_new_resolver("concat", lambda x, y: x + y)

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    text: Optional[str] = typer.Option(None, help="Text to be analyzed."),
    filename: Optional[Path] = typer.Option(None, help="File to be analyzed."),
    discourse: Optional[bool] = typer.Option(True, help="Whether to perform discourse relation analysis."),
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
    extended_vocab_path = resources.files("kwja") / "resource/typo_correction/multi_char_vocab.txt"
    typo_cfg.datamodule.predict.extended_vocab_path = str(extended_vocab_path)
    typo_cfg.dataset.extended_vocab_path = str(extended_vocab_path)
    typo_cfg.callbacks.prediction_writer.extended_vocab_path = str(extended_vocab_path)
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
    word_checkpoint = torch.load(str(word_checkpoint_path), map_location=lambda storage, loc: storage)
    hparams = word_checkpoint["hyper_parameters"]["hparams"]
    reading_resource_path = resources.files("kwja") / "resource/reading_prediction"
    hparams.datamodule.predict.reading_resource_path = reading_resource_path
    hparams.dataset.reading_resource_path = reading_resource_path
    hparams.callbacks.prediction_writer.reading_resource_path = reading_resource_path
    hparams.callbacks.prediction_writer.jumandic_path = resources.files("kwja") / "resource/jumandic"
    word_model: WordModule = WordModule.load_from_checkpoint(str(word_checkpoint_path), hparams=hparams)
    word_cfg = word_model.hparams
    word_cfg.datamodule.predict.reading_resource_path = reading_resource_path
    word_cfg.dataset.reading_resource_path = reading_resource_path
    word_cfg.callbacks.prediction_writer.reading_resource_path = reading_resource_path
    word_cfg.callbacks.prediction_writer.jumandic_path = resources.files("kwja") / "resource/jumandic"
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
    comments: list[str] = [x for i, x in enumerate(char_results) if i % 2 == 0]
    word_cfg.datamodule.predict.texts = [x for i, x in enumerate(char_results) if i % 2 == 1]
    word_datamodule = DataModule(cfg=word_cfg.datamodule)
    word_datamodule.setup(stage=TrainerFn.PREDICTING)
    word_trainer.predict(model=word_model, dataloaders=word_datamodule.predict_dataloader())
    word_module_writer: WordModuleWriter = word_trainer.callbacks[0]
    # word module (discourse) cannot be initialized unless this is written because multiple tinyDBs cannot be opened.
    word_module_writer.jumandic.close()
    del word_model
    document: Document = Document.from_knp(word_path.read_text())
    for idx, sentence in enumerate(document.sentences):
        sentence.comment = comments[idx]
    # Remove the result of discourse relation analysis by the jointly learned model.
    for base_phrase in document.base_phrases:
        if "談話関係" in base_phrase.features:
            del base_phrase.features["談話関係"]
    if not discourse:
        print(document.to_knp(), end="")
    else:
        # word module (discourse)
        word_discourse_checkpoint_path: Path = download_checkpoint_from_url(WORD_DISCOURSE_CHECKPOINT_URL)
        word_discourse_checkpoint = torch.load(
            str(word_discourse_checkpoint_path), map_location=lambda storage, loc: storage
        )
        hparams = word_discourse_checkpoint["hyper_parameters"]["hparams"]
        reading_resource_path = resources.files("kwja") / "resource/reading_prediction"
        hparams.datamodule.predict.reading_resource_path = reading_resource_path
        hparams.dataset.reading_resource_path = reading_resource_path
        hparams.callbacks.prediction_writer.reading_resource_path = reading_resource_path
        hparams.callbacks.prediction_writer.jumandic_path = resources.files("kwja") / "resource/jumandic"
        word_discourse_model: WordModule = WordModule.load_from_checkpoint(
            str(word_discourse_checkpoint_path), hparams=hparams
        )
        word_discourse_cfg = word_discourse_model.hparams
        word_discourse_cfg.datamodule.predict.reading_resource_path = reading_resource_path
        word_discourse_cfg.dataset.reading_resource_path = reading_resource_path

        word_discourse_trainer: pl.Trainer = pl.Trainer(
            logger=False,
            enable_progress_bar=False,
            callbacks=[
                WordModuleDiscourseWriter(
                    output_dir=str(tmp_dir.name),
                    pred_filename=word_discourse_path.stem,
                )
            ],
            devices=1,
        )
        word_discourse_cfg.datamodule.predict.knp_file = word_path
        word_discourse_datamodule = DataModule(cfg=word_discourse_cfg.datamodule)
        word_discourse_datamodule.setup(stage=TrainerFn.PREDICTING)
        word_discourse_trainer.predict(
            model=word_discourse_model,
            dataloaders=word_discourse_datamodule.predict_dataloader(),
        )
        discourse_document: Document = Document.from_knp(word_discourse_path.read_text())
        for idx, sentence in enumerate(discourse_document.sentences):
            sentence.comment = comments[idx]
        print(discourse_document.to_knp(), end="")
    tmp_dir.cleanup()


if __name__ == "__main__":
    typer.run(main)
