import os
import re
from abc import ABC
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Set, Tuple

import hydra
import pytorch_lightning as pl
import typer
from omegaconf import OmegaConf
from pytorch_lightning.trainer.states import TrainerFn
from rhoknp import Document
from rhoknp.utils.reader import chunk_by_document

import kwja
from kwja.cli.utils import download_checkpoint, prepare_device, suppress_debug_info
from kwja.datamodule.datamodule import DataModule
from kwja.modules import CharModule, Seq2SeqModule, TypoModule, WordModule

suppress_debug_info()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
OmegaConf.register_new_resolver("concat", lambda x, y: x + y)
OMEGACONF_VARIABLE_INTERPOLATION = re.compile(r"\$(?P<variable>\{.+?})")

app = typer.Typer(pretty_exceptions_show_locals=False)


class Device(str, Enum):
    auto = "auto"
    cpu = "cpu"
    gpu = "gpu"


class BaseModuleProcessor(ABC):
    def __init__(
        self,
        specified_device: str,
        model_size: str,
        batch_size: int,
        destination: Path,
    ) -> None:
        self.device_name, self.device = prepare_device(specified_device)
        self.model_size: str = model_size
        self.batch_size: int = batch_size
        self.destination: Path = destination
        self.module: Optional[pl.LightningModule] = None
        self.trainer: Optional[pl.Trainer] = None

    def load(self):
        self.module = self._load_module()
        self.module.hparams.datamodule.batch_size = self.batch_size

        writer = hydra.utils.instantiate(self.module.hparams.callbacks.prediction_writer, destination=self.destination)
        self.trainer = pl.Trainer(
            logger=False,
            callbacks=[writer, hydra.utils.instantiate(self.module.hparams.callbacks.progress_bar)],
            accelerator=self.device_name,
            devices=1,
        )

    def _load_module(self) -> pl.LightningModule:
        raise NotImplementedError

    def delete_module_and_trainer(self) -> None:
        del self.module, self.trainer

    def apply_module(self, input_file: Path) -> Path:
        datamodule = self._load_datamodule(input_file)
        assert self.trainer is not None
        self.trainer.predict(model=self.module, dataloaders=datamodule.predict_dataloader(), return_predictions=False)
        return self.destination

    def _load_datamodule(self, input_file: Path) -> DataModule:
        raise NotImplementedError

    def output_prediction(self) -> None:
        raise NotImplementedError


class TypoModuleProcessor(BaseModuleProcessor):
    def _load_module(self) -> pl.LightningModule:
        typer.echo("Loading typo module", err=True)
        checkpoint_path: Path = download_checkpoint(module="typo", model_size=self.model_size)
        return TypoModule.load_from_checkpoint(checkpoint_path, map_location=self.device)

    def _load_datamodule(self, input_file: Path) -> DataModule:
        assert self.module is not None
        self.module.hparams.datamodule.predict.texts = input_file.read_text().splitlines()
        datamodule = DataModule(cfg=self.module.hparams.datamodule)
        datamodule.setup(stage=TrainerFn.PREDICTING)
        return datamodule

    def output_prediction(self) -> None:
        post_texts: List[str] = []
        with self.destination.open() as f:
            for line in f:
                if line.strip() != "EOD":
                    post_texts.append(line.strip())
        print("\n".join(post_texts))


class Seq2SeqModuleProcessor(BaseModuleProcessor):
    def _load_module(self):
        typer.echo("Loading seq2seq module", err=True)
        checkpoint_path: Path = download_checkpoint(module="seq2seq", model_size=self.model_size)
        return Seq2SeqModule.load_from_checkpoint(checkpoint_path, map_location=self.device)

    def _load_datamodule(self, input_file: Path) -> DataModule:
        assert self.module is not None
        self.module.hparams.datamodule.predict.texts = input_file.read_text().splitlines()
        datamodule = DataModule(cfg=self.module.hparams.datamodule)
        datamodule.setup(stage=TrainerFn.PREDICTING)
        return datamodule

    def output_prediction(self) -> None:
        seq2seq_texts: List[str] = []
        with self.destination.open() as f:
            for line in f:
                if line.strip() != "EOD":
                    seq2seq_texts.append(line.strip())
        print("\n".join(seq2seq_texts))


class CharModuleProcessor(BaseModuleProcessor):
    def _load_module(self) -> pl.LightningModule:
        typer.echo("Loading char module", err=True)
        checkpoint_path: Path = download_checkpoint(module="char", model_size=self.model_size)
        return CharModule.load_from_checkpoint(checkpoint_path, map_location=self.device)

    def _load_datamodule(self, input_file: Path) -> DataModule:
        assert self.module is not None
        self.module.hparams.datamodule.predict.texts = input_file.read_text().splitlines()
        datamodule = DataModule(cfg=self.module.hparams.datamodule)
        datamodule.setup(stage=TrainerFn.PREDICTING)
        return datamodule

    def output_prediction(self) -> None:
        word_segmented_texts: List[str] = []
        with self.destination.open() as f:
            for juman_text in chunk_by_document(f):
                document = Document.from_jumanpp(juman_text)
                word_segmented_texts.append(" ".join(m.text for m in document.morphemes))
        print("\n".join(word_segmented_texts))


class WordModuleProcessor(BaseModuleProcessor):
    def _load_module(self) -> pl.LightningModule:
        typer.echo("Loading word module", err=True)
        checkpoint_path: Path = download_checkpoint(module="word", model_size=self.model_size)
        return WordModule.load_from_checkpoint(checkpoint_path, map_location=self.device)

    def _load_datamodule(self, input_file: Path) -> DataModule:
        assert self.module is not None
        self.module.hparams.datamodule.predict.juman_file = input_file
        datamodule = DataModule(cfg=self.module.hparams.datamodule)
        datamodule.setup(stage=TrainerFn.PREDICTING)
        return datamodule

    def output_prediction(self) -> None:
        print(self.destination.read_text(), end="")


class WordDiscourseModuleProcessor(BaseModuleProcessor):
    def _load_module(self) -> pl.LightningModule:
        typer.echo("Loading word_discourse module", err=True)
        checkpoint_path: Path = download_checkpoint(module="word_discourse", model_size=self.model_size)
        module = WordModule.load_from_checkpoint(checkpoint_path, map_location=self.device)
        module.hparams.callbacks.prediction_writer = {
            "_target_": "kwja.callbacks.word_discourse_module_writer.WordDiscourseModuleWriter"
        }
        return module

    def _load_datamodule(self, input_file: Path) -> DataModule:
        assert self.module is not None
        self.module.hparams.datamodule.predict.knp_file = input_file
        datamodule = DataModule(cfg=self.module.hparams.datamodule)
        datamodule.setup(stage=TrainerFn.PREDICTING)
        return datamodule

    def output_prediction(self) -> None:
        print(self.destination.read_text(), end="")


class CLIProcessor:
    def __init__(
        self,
        specified_device: str,
        model_size: str,
        typo_batch_size: int,
        seq2seq_batch_size: int,
        char_batch_size: int,
        word_batch_size: int,
    ) -> None:
        self.tmp_dir = TemporaryDirectory()
        self.raw_destination = self.tmp_dir.name / Path("raw_text.txt")
        typo_destination = self.tmp_dir.name / Path("typo_prediction.txt")
        seq2seq_destination = self.tmp_dir.name / Path("seq2seq_prediction.seq2seq")
        char_destination = self.tmp_dir.name / Path("char_prediction.juman")
        word_destination = self.tmp_dir.name / Path("word_prediction.knp")
        word_discourse_destination = self.tmp_dir.name / Path("word_discourse_prediction.knp")
        self.processors: Dict[str, BaseModuleProcessor] = {
            "typo": TypoModuleProcessor(specified_device, model_size, typo_batch_size, typo_destination),
            "seq2seq": Seq2SeqModuleProcessor(specified_device, model_size, seq2seq_batch_size, seq2seq_destination),
            "char": CharModuleProcessor(specified_device, model_size, char_batch_size, char_destination),
            "word": WordModuleProcessor(specified_device, model_size, word_batch_size, word_destination),
            "word_discourse": WordDiscourseModuleProcessor(
                specified_device, model_size, word_batch_size, word_discourse_destination
            ),
        }

    def load_modules(self, tasks: List[str]) -> None:
        for task in tasks:
            self.processors[task].load()

    def refresh(self) -> None:
        for processor in self.processors.values():
            processor.destination.unlink(missing_ok=True)

    def run(self, input_text: str, specified_tasks: List[str], interactive: bool = False) -> None:
        input_texts = _split_input_texts([input_text])
        self.raw_destination.write_text("\n".join(input_texts))  # TODO: consider using pickle
        input_file = self.raw_destination
        for specified_task in specified_tasks:
            processor = self.processors[specified_task]
            if interactive is False:
                processor.load()
            if specified_task in {"char", "seq2seq"} and "typo" in specified_tasks:
                # char and seq2seq module after typo module
                input_texts = _split_input_texts([input_file.read_text()])
                input_file.write_text("\n".join(input_texts))  # TODO: consider using pickle
            input_file = processor.apply_module(input_file)
            if interactive is False:
                processor.delete_module_and_trainer()
        self.processors[specified_tasks[-1]].output_prediction()


def _split_input_texts(input_texts: List[str]) -> List[str]:
    split_texts: List[str] = []
    split_text: str = ""
    for input_text in input_texts:
        input_text_with_eod: str = input_text.strip() + "\nEOD"
        for text in input_text_with_eod.split("\n"):
            if text == "EOD":
                # hydra.utils.instantiateを実行する際に文字列${...}を補間しようとするのを防ぐ
                normalized = OMEGACONF_VARIABLE_INTERPOLATION.sub(r"$␣\g<variable>", split_text)
                # "#"で始まる行がコメント行と誤認識されることを防ぐ
                normalized = normalized.replace("#", "♯").replace("＃", "♯")  # TODO: use "＃" instead of "♯"
                split_texts.append(normalized.rstrip())
                split_text = ""
            else:
                split_text += f"{text}\n"
    return split_texts


def version_callback(value: bool) -> None:
    if value is True:
        typer.echo(f"KWJA {kwja.__version__}")
        raise typer.Exit()


def model_size_callback(value: str) -> str:
    if value not in ("tiny", "base", "large"):
        raise typer.BadParameter("model size must be one of 'tiny', 'base', or 'large'")
    return value


def tasks_callback(value: str) -> str:
    tasks: List[str] = value.split(",")
    if len(tasks) == 0:
        raise typer.BadParameter("task must be specified")
    for task in tasks:
        if task not in ("typo", "seq2seq", "char", "word", "word_discourse"):
            raise typer.BadParameter("invalid task name is contained")
    valid_task_combinations: Set[Tuple[str, ...]] = {
        ("typo",),
        ("char",),
        ("seq2seq",),
        ("seq2seq", "typo"),
        ("seq2seq", "word"),
        ("seq2seq", "typo", "word"),
        ("seq2seq", "word", "word_discourse"),
        ("seq2seq", "typo", "word", "word_discourse"),
        ("char", "typo"),
        ("char", "word"),
        ("char", "typo", "word"),
        ("char", "word", "word_discourse"),
        ("char", "typo", "word", "word_discourse"),
    }
    sorted_task: Tuple[str, ...] = tuple(sorted(tasks))
    if sorted_task not in valid_task_combinations:
        raise typer.BadParameter(
            "task combination is invalid. "
            "Please specify one of "
            "'typo', "
            "'char', "
            "'seq2seq', "
            "'typo,char', "
            "'typo,seq2seq', "
            "'seq2seq,word', "
            "'char,word', "
            "'typo,seq2seq,word', "
            "'seq2seq,word,word_discourse', "
            "'typo,seq2seq,word,word_discourse', "
            "'typo,char,word', "
            "'char,word,word_discourse' "
            "or 'typo,char,word,word_discourse'."
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
    seq2seq_batch_size: int = typer.Option(1, help="Batch size for seq2seq module."),
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
        input_text = Path(filename).read_text()

    if model_size == "large" and "seq2seq" in tasks:
        typer.echo("ERROR: Large model does not support seq2seq module now", err=True)
        raise typer.Abort()

    processor = CLIProcessor(
        specified_device=device.value,
        model_size=model_size,
        typo_batch_size=typo_batch_size,
        seq2seq_batch_size=seq2seq_batch_size,
        char_batch_size=char_batch_size,
        word_batch_size=word_batch_size,
    )
    specified_tasks: List[str] = [
        task for task in ["typo", "seq2seq", "char", "word", "word_discourse"] if task in tasks.split(",")
    ]

    if input_text is None:
        processor.load_modules(specified_tasks)

        typer.echo('Please end your input with a new line and type "EOD"', err=True)
        input_text = ""
        while True:
            input_ = input()
            if input_ == "EOD":
                processor.refresh()
                processor.run(input_text, specified_tasks, interactive=True)
                print("EOD")  # To indicate the end of the output.
                input_text = ""
            else:
                input_text += input_ + "\n"
    else:
        if input_text.strip() == "":
            raise typer.Exit()
        processor.run(input_text, specified_tasks)


if __name__ == "__main__":
    typer.run(main)
