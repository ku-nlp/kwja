import os
import re
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

import hydra
import pytorch_lightning as pl
import typer
from omegaconf import OmegaConf
from pytorch_lightning.trainer.states import TrainerFn
from rhoknp import Document
from rhoknp.utils.reader import chunk_by_document

import kwja
from kwja.callbacks.word_discourse_module_writer import WordDiscourseModuleWriter
from kwja.cli.utils import download_checkpoint, prepare_device, suppress_debug_info
from kwja.datamodule.datamodule import DataModule
from kwja.models.char_module import CharModule
from kwja.models.typo_module import TypoModule
from kwja.models.word_module import WordModule

suppress_debug_info()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
OmegaConf.register_new_resolver("concat", lambda x, y: x + y)
OMEGACONF_VARIABLE_INTERPOLATION = re.compile(r"\$(?P<variable>\{.+?})")

app = typer.Typer(pretty_exceptions_show_locals=False)


class Device(str, Enum):
    auto = "auto"
    cpu = "cpu"
    gpu = "gpu"


class Tasks:
    def __init__(self, tasks: List[str]):
        self.typo = "typo" in tasks
        self.char = "char" in tasks
        self.word = "word" in tasks
        self.word_discourse = "word_discourse" in tasks


class CLIProcessor:
    def __init__(
        self,
        specified_device: str,
        specified_tasks: Tasks,
        model_size: str,
        typo_batch_size: int,
        char_batch_size: int,
        word_batch_size: int,
        interactive: bool = False,
    ) -> None:
        self.device_name, self.device = prepare_device(specified_device)
        self.specified_tasks = specified_tasks
        self.model_size = model_size
        self.typo_batch_size = typo_batch_size
        self.char_batch_size = char_batch_size
        self.word_batch_size = word_batch_size
        self.interactive = interactive

        self.tmp_dir: TemporaryDirectory = TemporaryDirectory()
        self.typo_destination: Path = self.tmp_dir.name / Path("typo_prediction.txt")
        self.char_destination: Path = self.tmp_dir.name / Path("char_prediction.juman")
        self.word_destination: Path = self.tmp_dir.name / Path("word_prediction.knp")
        self.word_discourse_destination: Path = self.tmp_dir.name / Path("word_discourse_prediction.knp")

        self.typo_module: Optional[TypoModule] = None
        self.typo_trainer: Optional[pl.Trainer] = None
        self.char_module: Optional[CharModule] = None
        self.char_trainer: Optional[pl.Trainer] = None
        self.word_module: Optional[WordModule] = None
        self.word_trainer: Optional[pl.Trainer] = None
        self.word_discourse_module: Optional[WordModule] = None
        self.word_discourse_trainer: Optional[pl.Trainer] = None

        if self.interactive is True:
            if self.specified_tasks.typo is True:
                self.load_typo_module()
            if self.specified_tasks.char is True:
                self.load_char_module()
            if self.specified_tasks.word is True:
                self.load_word_module()
            if self.specified_tasks.word_discourse is True:
                self.load_word_discourse_module()

    @staticmethod
    def _split_input_texts(input_texts: List[str]) -> List[str]:
        split_texts: List[str] = []
        split_text: str = ""
        for input_text in input_texts:
            stripped_input_text: str = input_text.strip()
            if stripped_input_text.endswith("EOD"):
                stripped_input_text = stripped_input_text[:-3]
            input_text_with_eod: str = stripped_input_text + "\nEOD"
            for text in input_text_with_eod.split("\n"):
                if text == "EOD":
                    # hydra.utils.instantiateを実行する際に文字列${...}を補間しようとするのを防ぐ
                    normalized = OMEGACONF_VARIABLE_INTERPOLATION.sub(r"$␣\g<variable>", split_text)
                    # "#"で始まる行がコメント行と誤認識されることを防ぐ
                    normalized = normalized.replace("#", "♯")
                    split_texts.append(normalized.rstrip())
                    split_text = ""
                else:
                    split_text += f"{text}\n"
        return split_texts

    def load_typo_module(self) -> None:
        typer.echo("Loading typo module", err=True)
        typo_checkpoint_path: Path = download_checkpoint(task="typo", model_size=self.model_size)
        self.typo_module = TypoModule.load_from_checkpoint(str(typo_checkpoint_path), map_location=self.device)
        assert self.typo_module is not None, "typo module doesn't exist"
        self.typo_module.hparams.datamodule.batch_size = self.typo_batch_size
        self.typo_trainer = pl.Trainer(
            logger=False,
            callbacks=[
                hydra.utils.instantiate(
                    self.typo_module.hparams.callbacks.prediction_writer,
                    destination=self.typo_destination,
                ),
                hydra.utils.instantiate(self.typo_module.hparams.callbacks.progress_bar),
            ],
            accelerator=self.device_name,
            devices=1,
        )
        assert self.typo_trainer is not None, "typo trainer doesn't exist"

    def delete_typo_module(self) -> None:
        del self.typo_module, self.typo_trainer

    def apply_typo_module(self, input_texts: List[str]) -> None:
        if self.interactive is False:
            self.load_typo_module()
        assert self.typo_module is not None, "typo module doesn't exist"
        self.typo_module.hparams.datamodule.predict.texts = self._split_input_texts(input_texts)
        typo_datamodule = DataModule(cfg=self.typo_module.hparams.datamodule)
        typo_datamodule.setup(stage=TrainerFn.PREDICTING)
        assert self.typo_trainer is not None, "typo trainer doesn't exist"
        self.typo_trainer.predict(
            model=self.typo_module, dataloaders=typo_datamodule.predict_dataloader(), return_predictions=False
        )
        if self.interactive is False:
            self.delete_typo_module()

    def output_typo_prediction(self) -> None:
        typo_texts: List[str] = []
        with self.typo_destination.open(mode="r") as f:
            for line in f:
                if line.strip() != "EOD":
                    typo_texts.append(line.strip())
        print("\n".join(typo_texts))

    def load_char_module(self) -> None:
        typer.echo("Loading char module", err=True)
        char_checkpoint_path: Path = download_checkpoint(task="char", model_size=self.model_size)
        self.char_module = CharModule.load_from_checkpoint(str(char_checkpoint_path), map_location=self.device)
        assert self.char_module is not None, "char module doesn't exist"
        self.char_module.hparams.datamodule.batch_size = self.char_batch_size
        self.char_trainer = pl.Trainer(
            logger=False,
            callbacks=[
                hydra.utils.instantiate(
                    self.char_module.hparams.callbacks.prediction_writer,
                    destination=self.char_destination,
                ),
                hydra.utils.instantiate(self.char_module.hparams.callbacks.progress_bar),
            ],
            accelerator=self.device_name,
            devices=1,
        )
        assert self.char_trainer is not None, "char trainer doesn't exist"

    def delete_char_module(self) -> None:
        del self.char_module, self.char_trainer

    def apply_char_module(self, input_texts: Optional[List[str]] = None) -> None:
        if self.interactive is False:
            self.load_char_module()
        assert self.char_module is not None, "char module doesn't exist"
        if input_texts is None:
            self.char_module.hparams.datamodule.predict.texts = self._split_input_texts(
                [self.typo_destination.read_text()]
            )
        else:
            self.char_module.hparams.datamodule.predict.texts = self._split_input_texts(input_texts)
        char_datamodule = DataModule(cfg=self.char_module.hparams.datamodule)
        char_datamodule.setup(stage=TrainerFn.PREDICTING)
        assert self.char_trainer is not None, "char trainer doesn't exist"
        self.char_trainer.predict(
            model=self.char_module, dataloaders=char_datamodule.predict_dataloader(), return_predictions=False
        )
        if self.interactive is False:
            self.delete_char_module()

    def output_char_prediction(self) -> None:
        word_segmented_texts: List[str] = []
        with self.char_destination.open(mode="r") as f:
            for juman_text in chunk_by_document(f):
                document = Document.from_jumanpp(juman_text)
                word_segmented_texts.append(" ".join(m.text for m in document.morphemes))
        print("\n".join(word_segmented_texts))

    def load_word_module(self) -> None:
        typer.echo("Loading word module", err=True)
        word_checkpoint_path: Path = download_checkpoint(task="word", model_size=self.model_size)
        self.word_module = WordModule.load_from_checkpoint(str(word_checkpoint_path), map_location=self.device)
        assert self.word_module is not None, "word module doesn't exist"
        self.word_module.hparams.datamodule.batch_size = self.word_batch_size
        self.word_trainer = pl.Trainer(
            logger=False,
            callbacks=[
                hydra.utils.instantiate(
                    self.word_module.hparams.callbacks.prediction_writer,
                    destination=self.word_destination,
                ),
                hydra.utils.instantiate(self.word_module.hparams.callbacks.progress_bar),
            ],
            accelerator=self.device_name,
            devices=1,
        )
        assert self.word_trainer is not None, "word trainer doesn't exist"

    def delete_word_module(self) -> None:
        del self.word_module, self.word_trainer

    def apply_word_module(self) -> None:
        if self.interactive is False:
            self.load_word_module()
        assert self.word_module is not None, "word module doesn't exist"
        self.word_module.hparams.datamodule.predict.juman_file = self.char_destination
        word_datamodule = DataModule(cfg=self.word_module.hparams.datamodule)
        word_datamodule.setup(stage=TrainerFn.PREDICTING)
        assert self.word_trainer is not None, "word trainer doesn't exist"
        self.word_trainer.predict(
            model=self.word_module, dataloaders=word_datamodule.predict_dataloader(), return_predictions=False
        )
        if self.interactive is False:
            self.delete_word_module()

    def output_word_prediction(self) -> None:
        print(self.word_destination.read_text(), end="")

    def load_word_discourse_module(self) -> None:
        typer.echo("Loading word discourse module", err=True)
        word_discourse_checkpoint_path: Path = download_checkpoint(task="word_discourse", model_size=self.model_size)
        self.word_discourse_module = WordModule.load_from_checkpoint(
            str(word_discourse_checkpoint_path), map_location=self.device
        )
        assert self.word_discourse_module is not None, "word discourse module doesn't exist"
        self.word_discourse_module.hparams.datamodule.batch_size = self.word_batch_size
        self.word_discourse_trainer = pl.Trainer(
            logger=False,
            callbacks=[
                WordDiscourseModuleWriter(destination=self.word_discourse_destination),
                hydra.utils.instantiate(self.word_discourse_module.hparams.callbacks.progress_bar),
            ],
            accelerator=self.device_name,
            devices=1,
        )
        assert self.word_discourse_trainer is not None, "word discourse trainer doesn't exist"

    def delete_word_discourse_module(self) -> None:
        del self.word_discourse_module, self.word_discourse_trainer

    def apply_word_discourse_module(self) -> None:
        if self.interactive is False:
            self.load_word_discourse_module()
        assert self.word_discourse_module is not None, "word discourse module doesn't exist"
        self.word_discourse_module.hparams.datamodule.predict.knp_file = self.word_destination
        word_discourse_datamodule = DataModule(cfg=self.word_discourse_module.hparams.datamodule)
        word_discourse_datamodule.setup(stage=TrainerFn.PREDICTING)
        assert self.word_discourse_trainer is not None, "word discourse trainer doesn't exist"
        self.word_discourse_trainer.predict(
            model=self.word_discourse_module,
            dataloaders=word_discourse_datamodule.predict_dataloader(),
            return_predictions=False,
        )
        if self.interactive is False:
            self.delete_word_discourse_module()

    def output_word_discourse_prediction(self) -> None:
        print(self.word_discourse_destination.read_text(), end="")

    def run(self, input_text: str) -> None:
        self.typo_destination.unlink(missing_ok=True)
        self.char_destination.unlink(missing_ok=True)
        self.word_destination.unlink(missing_ok=True)
        self.word_discourse_destination.unlink(missing_ok=True)

        if self.specified_tasks.typo is True:
            self.apply_typo_module([input_text])
            if self.specified_tasks.char is False:
                self.output_typo_prediction()
        if self.specified_tasks.char is True:
            self.apply_char_module(input_texts=[input_text] if self.specified_tasks.typo is False else None)
            if self.specified_tasks.word is False:
                self.output_char_prediction()
        if self.specified_tasks.word is True:
            self.apply_word_module()
            if self.specified_tasks.word_discourse is False:
                self.output_word_prediction()
        if self.specified_tasks.word_discourse is True:
            self.apply_word_discourse_module()
            self.output_word_discourse_prediction()


def version_callback(value: bool) -> None:
    if value is True:
        typer.echo(f"KWJA {kwja.__version__}")
        raise typer.Exit()


def model_size_callback(value: str) -> str:
    if value not in ["tiny", "base", "large"]:
        raise typer.BadParameter("model size must be one of 'tiny', 'base', or 'large'")
    return value


def tasks_callback(value: str) -> str:
    tasks: List[str] = value.split(",")
    if len(tasks) == 0:
        raise typer.BadParameter("task must be specified")
    for task in tasks:
        if task not in ["typo", "char", "word", "word_discourse"]:
            raise typer.BadParameter("invalid task name is contained")
    valid_task_combinations: List[List[str]] = [
        ["typo"],
        ["char"],
        ["char", "typo"],
        ["char", "word"],
        ["char", "typo", "word"],
        ["char", "word", "word_discourse"],
        ["char", "typo", "word", "word_discourse"],
    ]
    sorted_task: List[str] = sorted(tasks)
    if sorted_task not in valid_task_combinations:
        raise typer.BadParameter(
            "task combination is invalid. Please specify one of 'typo', 'char', 'typo,char', 'char,word', 'typo,char,word', 'char,word,word_discourse' or 'typo,char,word,word_discourse'"
        )
    return value


@app.command()
def main(
    text: Optional[str] = typer.Option(None, help="Text to be analyzed."),
    filename: Optional[Path] = typer.Option(None, help="File to be analyzed."),
    device: Device = typer.Option(
        Device.auto,
        help="Device to be used. Please specify 'auto', 'cpu' or 'gpu'.",
    ),
    model_size: str = typer.Option(
        "base", callback=model_size_callback, help="Model size to be used. Please specify 'tiny', 'base', or 'large'."
    ),
    typo_batch_size: int = typer.Option(1, help="Batch size for typo module."),
    char_batch_size: int = typer.Option(1, help="Batch size for char module."),
    word_batch_size: int = typer.Option(1, help="Batch size for word module."),
    tasks: str = typer.Option("typo,char,word,word_discourse", callback=tasks_callback, help="Tasks to be performed."),
    _: Optional[bool] = typer.Option(None, "--version", callback=version_callback, is_eager=True),
) -> None:
    input_text: Optional[str] = None
    if text is not None and filename is not None:
        typer.echo("ERROR: Please provide text or filename, not both", err=True)
        raise typer.Abort()
    elif text is not None:
        input_text = text
    elif filename is not None:
        with Path(filename).open() as f:
            input_text = f.read()

    processor: CLIProcessor = CLIProcessor(
        specified_device=device.value,
        specified_tasks=Tasks(tasks.split(",")),
        model_size=model_size,
        typo_batch_size=typo_batch_size,
        char_batch_size=char_batch_size,
        word_batch_size=word_batch_size,
        interactive=input_text is None,
    )

    if input_text is None:
        typer.echo('Please end your input with a new line and type "EOD"', err=True)
        input_text = ""
        while True:
            input_ = input()
            if input_ == "EOD":
                processor.run(input_text)
                print("EOD")  # To indicate the end of the output.
                input_text = ""
            else:
                input_text += input_ + "\n"
    else:
        if input_text.strip() == "":
            raise typer.Exit()
        processor.run(input_text)


if __name__ == "__main__":
    typer.run(main)
