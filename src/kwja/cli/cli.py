import os
import re
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

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
from kwja.modules import CharModule, TypoModule, WordModule

suppress_debug_info()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
OmegaConf.register_new_resolver("concat", lambda x, y: x + y)
OMEGACONF_VARIABLE_INTERPOLATION = re.compile(r"\$(?P<variable>\{.+?})")

app = typer.Typer(pretty_exceptions_show_locals=False)


class Device(str, Enum):
    auto = "auto"
    cpu = "cpu"
    gpu = "gpu"


class CLIProcessor:
    def __init__(
        self,
        specified_device: str,
        model_size: str,
        typo_batch_size: int,
        char_batch_size: int,
        word_batch_size: int,
    ) -> None:
        self.device_name, self.device = prepare_device(specified_device)
        self.model_size = model_size

        self.tmp_dir: TemporaryDirectory = TemporaryDirectory()
        self.task2task_settings: Dict[str, Any] = {
            "typo": {
                "module_class": TypoModule,
                "batch_size": typo_batch_size,
                "destination": self.tmp_dir.name / Path("typo_prediction.txt"),
            },
            "char": {
                "module_class": CharModule,
                "batch_size": char_batch_size,
                "destination": self.tmp_dir.name / Path("char_prediction.juman"),
            },
            "word": {
                "module_class": WordModule,
                "batch_size": word_batch_size,
                "destination": self.tmp_dir.name / Path("word_prediction.knp"),
            },
            "word_discourse": {
                "module_class": WordModule,
                "batch_size": word_batch_size,
                "destination": self.tmp_dir.name / Path("word_discourse_prediction.knp"),
            },
        }

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

    def load_module(self, task: str) -> None:
        typer.echo(f"Loading {task} module", err=True)

        checkpoint_path: Path = download_checkpoint(task=task, model_size=self.model_size)
        module = self.task2task_settings[task]["module_class"].load_from_checkpoint(
            str(checkpoint_path), map_location=self.device
        )
        module.hparams.datamodule.batch_size = self.task2task_settings[task]["batch_size"]
        self.task2task_settings[task]["module"] = module

        if task == "word_discourse":
            module.hparams.callbacks.prediction_writer = {
                "_target_": "kwja.callbacks.word_discourse_module_writer.WordDiscourseModuleWriter"
            }
        writer = hydra.utils.instantiate(
            module.hparams.callbacks.prediction_writer, destination=self.task2task_settings[task]["destination"]
        )
        trainer = pl.Trainer(
            logger=False,
            callbacks=[writer, hydra.utils.instantiate(module.hparams.callbacks.progress_bar)],
            accelerator=self.device_name,
            devices=1,
        )
        self.task2task_settings[task]["trainer"] = trainer

    def delete_module_and_trainer(self, task: str) -> None:
        del self.task2task_settings[task]["module"], self.task2task_settings[task]["trainer"]

    def apply_module(self, task: str, input_texts: Optional[List[str]] = None):
        module = self.task2task_settings[task]["module"]
        if input_texts is not None:
            module.hparams.datamodule.predict.texts = self._split_input_texts(input_texts)
        else:
            if task == "char":
                module.hparams.datamodule.predict.texts = self._split_input_texts(
                    [self.task2task_settings["typo"]["destination"].read_text()]
                )
            elif task == "word":
                module.hparams.datamodule.predict.juman_file = self.task2task_settings["char"]["destination"]
            elif task == "word_discourse":
                module.hparams.datamodule.predict.knp_file = self.task2task_settings["word"]["destination"]

        datamodule = DataModule(cfg=module.hparams.datamodule)
        datamodule.setup(stage=TrainerFn.PREDICTING)

        trainer = self.task2task_settings[task]["trainer"]
        trainer.predict(model=module, dataloaders=datamodule.predict_dataloader(), return_predictions=False)

    def output_prediction(self, task: str) -> None:
        if task == "typo":
            post_texts: List[str] = []
            with self.task2task_settings[task]["destination"].open(mode="r") as f:
                for line in f:
                    if line.strip() != "EOD":
                        post_texts.append(line.strip())
            print("\n".join(post_texts))
        elif task == "char":
            word_segmented_texts: List[str] = []
            with self.task2task_settings[task]["destination"].open(mode="r") as f:
                for juman_text in chunk_by_document(f):
                    document = Document.from_jumanpp(juman_text)
                    word_segmented_texts.append(" ".join(m.text for m in document.morphemes))
            print("\n".join(word_segmented_texts))
        else:
            print(self.task2task_settings[task]["destination"].read_text(), end="")

    def refresh(self):
        for task_settings in self.task2task_settings.values():
            task_settings["destination"].unlink(missing_ok=True)

    def run(self, input_text: str, specified_tasks: List[str], interactive: bool = False) -> None:
        for specified_task in specified_tasks:
            if interactive is False:
                self.load_module(specified_task)
            if (specified_task == "typo") or (specified_task == "char" and "typo" not in specified_tasks):
                input_texts = [input_text]
            else:
                input_texts = None
            self.apply_module(specified_task, input_texts=input_texts)
            if interactive is False:
                self.delete_module_and_trainer(specified_task)
        self.output_prediction(specified_tasks[-1])


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
            "task combination is invalid. "
            "Please specify one of "
            "'typo', "
            "'char', "
            "'typo,char', "
            "'char,word', "
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

    processor = CLIProcessor(
        specified_device=device.value,
        model_size=model_size,
        typo_batch_size=typo_batch_size,
        char_batch_size=char_batch_size,
        word_batch_size=word_batch_size,
    )
    specified_tasks: List[str] = [
        task for task in ["typo", "char", "word", "word_discourse"] if task in tasks.split(",")
    ]

    if input_text is None:
        for specified_task in specified_tasks:
            processor.load_module(specified_task)

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
