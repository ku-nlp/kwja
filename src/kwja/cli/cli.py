from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

import hydra
import importlib_resources
import pytorch_lightning as pl
import torch
import typer
from omegaconf import OmegaConf
from pytorch_lightning.trainer.states import TrainerFn
from rhoknp import Document

import kwja
from kwja.callbacks.word_module_discourse_writer import WordModuleDiscourseWriter
from kwja.cli.utils import download_checkpoint_from_url, prepare_device, suppress_debug_info
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
resource_path = importlib_resources.files(kwja) / "resource"


class CLIProcessor:
    def __init__(self, specified_device: str) -> None:
        self.device_name, self.device = prepare_device(specified_device)

        self.tmp_dir: TemporaryDirectory = TemporaryDirectory()
        self.typo_path: Path = self.tmp_dir.name / Path("predict_typo.txt")
        self.char_path: Path = self.tmp_dir.name / Path("predict_char.juman")
        self.word_path: Path = self.tmp_dir.name / Path("predict_word.knp")
        self.word_discourse_path: Path = self.tmp_dir.name / Path("predict_word_discourse.knp")

        self.typo_model: Optional[TypoModule] = None
        self.typo_trainer: Optional[pl.Trainer] = None
        self.char_model: Optional[CharModule] = None
        self.char_trainer: Optional[pl.Trainer] = None
        self.word_model: Optional[WordModule] = None
        self.word_trainer: Optional[pl.Trainer] = None
        self.word_discourse_model: Optional[WordModule] = None
        self.word_discourse_trainer: Optional[pl.Trainer] = None

    def load_typo(self) -> None:
        typo_checkpoint_path: Path = download_checkpoint_from_url(TYPO_CHECKPOINT_URL)
        self.typo_model = TypoModule.load_from_checkpoint(
            str(typo_checkpoint_path),
            map_location=self.device,
        )
        extended_vocab_path = resource_path / "typo_correction/multi_char_vocab.txt"
        if self.typo_model is None:
            raise ValueError("typo model does not exist")
        self.typo_model.hparams.datamodule.predict.extended_vocab_path = str(extended_vocab_path)
        self.typo_model.hparams.dataset.extended_vocab_path = str(extended_vocab_path)
        self.typo_model.hparams.callbacks.prediction_writer.extended_vocab_path = str(extended_vocab_path)
        self.typo_trainer = pl.Trainer(
            logger=False,
            enable_progress_bar=False,
            callbacks=[
                hydra.utils.instantiate(
                    self.typo_model.hparams.callbacks.prediction_writer,
                    output_dir=str(self.tmp_dir.name),
                    pred_filename=self.typo_path.stem,
                )
            ],
            accelerator=self.device_name,
            devices=1,
        )

    def apply_typo(self, input_texts: List[str]) -> None:
        if self.typo_model is None:
            raise ValueError("typo model does not exist")
        self.typo_model.hparams.datamodule.predict.texts = input_texts
        typo_datamodule = DataModule(cfg=self.typo_model.hparams.datamodule)
        typo_datamodule.setup(stage=TrainerFn.PREDICTING)
        if self.typo_trainer is None:
            raise ValueError("typo trainer does not exist")
        self.typo_trainer.predict(model=self.typo_model, dataloaders=typo_datamodule.predict_dataloader())

    def del_typo(self) -> None:
        del self.typo_model, self.typo_trainer

    def load_char(self) -> None:
        char_checkpoint_path: Path = download_checkpoint_from_url(CHAR_CHECKPOINT_URL)
        self.char_model = CharModule.load_from_checkpoint(
            str(char_checkpoint_path),
            map_location=self.device,
        )
        if self.char_model is None:
            raise ValueError("char model does not exist")
        self.char_trainer = pl.Trainer(
            logger=False,
            enable_progress_bar=False,
            callbacks=[
                hydra.utils.instantiate(
                    self.char_model.hparams.callbacks.prediction_writer,
                    output_dir=str(self.tmp_dir.name),
                    pred_filename=self.char_path.stem,
                )
            ],
            accelerator=self.device_name,
            devices=1,
        )

    def apply_char(self) -> None:
        if self.char_model is None:
            raise ValueError("char model does not exist")
        self.char_model.hparams.datamodule.predict.texts = self.typo_path.read_text().splitlines()
        char_datamodule = DataModule(cfg=self.char_model.hparams.datamodule)
        char_datamodule.setup(stage=TrainerFn.PREDICTING)
        if self.char_trainer is None:
            raise ValueError("char trainer does not exist")
        self.char_trainer.predict(model=self.char_model, dataloaders=char_datamodule.predict_dataloader())

    def del_char(self) -> None:
        del self.char_model, self.char_trainer

    def load_word(self) -> None:
        word_checkpoint_path: Path = download_checkpoint_from_url(WORD_CHECKPOINT_URL)
        word_checkpoint = torch.load(str(word_checkpoint_path), map_location=lambda storage, loc: storage)
        hparams = word_checkpoint["hyper_parameters"]["hparams"]
        reading_resource_path = resource_path / "reading_prediction"
        jumandic_path = resource_path / "jumandic"
        hparams.datamodule.predict.reading_resource_path = reading_resource_path
        hparams.dataset.reading_resource_path = reading_resource_path
        hparams.callbacks.prediction_writer.reading_resource_path = reading_resource_path
        hparams.callbacks.prediction_writer.jumandic_path = jumandic_path
        self.word_model = WordModule.load_from_checkpoint(
            str(word_checkpoint_path),
            hparams=hparams,
            map_location=self.device,
        )
        if self.word_model is None:
            raise ValueError("word model does not exist")
        self.word_model.hparams.datamodule.predict.reading_resource_path = reading_resource_path
        self.word_model.hparams.dataset.reading_resource_path = reading_resource_path
        self.word_model.hparams.callbacks.prediction_writer.reading_resource_path = reading_resource_path
        self.word_model.hparams.callbacks.prediction_writer.jumandic_path = jumandic_path
        self.word_trainer = pl.Trainer(
            logger=False,
            enable_progress_bar=False,
            callbacks=[
                hydra.utils.instantiate(
                    self.word_model.hparams.callbacks.prediction_writer,
                    output_dir=str(self.tmp_dir.name),
                    pred_filename=self.word_path.stem,
                )
            ],
            accelerator=self.device_name,
            devices=1,
        )

    def apply_word(self) -> None:
        if self.word_model is None:
            raise ValueError("word model does not exist")
        self.word_model.hparams.datamodule.predict.juman_file = self.char_path
        word_datamodule = DataModule(cfg=self.word_model.hparams.datamodule)
        word_datamodule.setup(stage=TrainerFn.PREDICTING)
        if self.word_trainer is None:
            raise ValueError("word trainer does not exist")
        self.word_trainer.predict(model=self.word_model, dataloaders=word_datamodule.predict_dataloader())

    def del_word(self) -> None:
        del self.word_model, self.word_trainer

    def load_word_discourse(self) -> None:
        word_discourse_checkpoint_path: Path = download_checkpoint_from_url(WORD_DISCOURSE_CHECKPOINT_URL)
        word_discourse_checkpoint = torch.load(
            str(word_discourse_checkpoint_path), map_location=lambda storage, loc: storage
        )
        hparams = word_discourse_checkpoint["hyper_parameters"]["hparams"]
        reading_resource_path = resource_path / "reading_prediction"
        jumandic_path = resource_path / "jumandic"
        hparams.datamodule.predict.reading_resource_path = reading_resource_path
        hparams.dataset.reading_resource_path = reading_resource_path
        hparams.callbacks.prediction_writer.reading_resource_path = reading_resource_path
        hparams.callbacks.prediction_writer.jumandic_path = jumandic_path
        self.word_discourse_model = WordModule.load_from_checkpoint(
            str(word_discourse_checkpoint_path),
            hparams=hparams,
            map_location=self.device,
        )
        if self.word_discourse_model is None:
            raise ValueError("word discourse model does not exist")
        self.word_discourse_model.hparams.datamodule.predict.reading_resource_path = reading_resource_path
        self.word_discourse_model.hparams.dataset.reading_resource_path = reading_resource_path
        self.word_discourse_trainer = pl.Trainer(
            logger=False,
            enable_progress_bar=False,
            callbacks=[
                WordModuleDiscourseWriter(
                    output_dir=str(self.tmp_dir.name),
                    pred_filename=self.word_discourse_path.stem,
                )
            ],
            accelerator=self.device_name,
            devices=1,
        )

    def apply_word_discourse(self) -> None:
        if self.word_discourse_model is None:
            raise ValueError("word discourse model does not exist")
        self.word_discourse_model.hparams.datamodule.predict.knp_file = self.word_path
        word_discourse_datamodule = DataModule(cfg=self.word_discourse_model.hparams.datamodule)
        word_discourse_datamodule.setup(stage=TrainerFn.PREDICTING)
        if self.word_discourse_trainer is None:
            raise ValueError("word discourse trainer does not exist")
        self.word_discourse_trainer.predict(
            model=self.word_discourse_model,
            dataloaders=word_discourse_datamodule.predict_dataloader(),
        )

    def output_word_result(self) -> None:
        document: Document = Document.from_knp(self.word_path.read_text())
        # Remove the result of discourse relation analysis by the jointly learned model.
        for base_phrase in document.base_phrases:
            if "談話関係" in base_phrase.features:
                del base_phrase.features["談話関係"]
        print(document.to_knp(), end="")

    def output_word_discourse_result(self) -> None:
        discourse_document: Document = Document.from_knp(self.word_discourse_path.read_text())
        print(discourse_document.to_knp(), end="")


def version_callback(value: bool) -> None:
    if value is True:
        typer.echo(f"KWJA {kwja.__version__}")
        raise typer.Exit()


@app.command()
def main(
    text: Optional[str] = typer.Option(None, help="Text to be analyzed."),
    filename: Optional[Path] = typer.Option(None, help="File to be analyzed."),
    device: str = typer.Option("cpu", help="Device to be used. Please specify 'cpu' or 'gpu'."),
    discourse: Optional[bool] = typer.Option(True, help="Whether to perform discourse relation analysis."),
    _: Optional[bool] = typer.Option(None, "--version", callback=version_callback, is_eager=True),
) -> None:
    input_text = ""
    if text is not None and filename is not None:
        typer.echo("ERROR: Please provide text or filename, not both", err=True)
        raise typer.Exit()
    elif text is not None:
        input_text = text
    elif filename is not None:
        with Path(filename).open() as f:
            input_text = f.read()

    processor = CLIProcessor(specified_device=device)
    if input_text:
        processor.load_typo()
        processor.apply_typo([input_text])
        processor.del_typo()

        processor.load_char()
        processor.apply_char()
        processor.del_char()

        processor.load_word()
        processor.apply_word()
        processor.del_word()

        if not discourse:
            processor.output_word_result()
        else:
            processor.load_word_discourse()
            processor.apply_word_discourse()
            processor.output_word_discourse_result()
    else:
        typer.echo('Please end your input with a new line and type "EOD"', err=True)
        processor.load_typo()
        processor.load_char()
        processor.load_word()
        if discourse:
            processor.load_word_discourse()

        while True:
            inp = input()
            if inp == "EOD":
                processor.apply_typo([input_text])
                processor.apply_char()
                processor.apply_word()
                if not discourse:
                    processor.output_word_result()
                    print("EOD")  # To indicate the end of the output.
                else:
                    processor.apply_word_discourse()
                    processor.output_word_discourse_result()
                    print("EOD")  # To indicate the end of the output.
                input_text = ""
            else:
                input_text += inp + "\n"


if __name__ == "__main__":
    typer.run(main)
