import os
import re
import sys
from abc import ABC
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Set, Tuple
from unicodedata import normalize

import hydra
import pytorch_lightning as pl
import torch
import typer
from pytorch_lightning.trainer.states import TrainerFn
from rhoknp import RegexSenter, Sentence
from rhoknp.utils.reader import chunk_by_sentence

import kwja
from kwja.cli.config import CLIConfig, Device, ModelSize, get_kwja_config_file
from kwja.cli.utils import download_checkpoint, prepare_device
from kwja.datamodule.datamodule import DataModule
from kwja.modules import CharModule, SenterModule, Seq2SeqModule, TypoModule, WordModule
from kwja.utils.constants import TRANSLATION_TABLE
from kwja.utils.logging_util import filter_logs

filter_logs(environment="production")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
OMEGACONF_VARIABLE_INTERPOLATION = re.compile(r"\$(?P<variable>\{.+?})")

app = typer.Typer(pretty_exceptions_show_locals=False)


class BaseModuleProcessor(ABC):
    def __init__(self, config: CLIConfig, batch_size: int) -> None:
        self.config: CLIConfig = config
        self.device_name, self.device = prepare_device(config.device.value)
        self.model_size: ModelSize = config.model_size
        self.batch_size: int = batch_size
        self.destination = Path(NamedTemporaryFile().name)
        self.module: Optional[pl.LightningModule] = None
        self.trainer: Optional[pl.Trainer] = None

    def load(self, **writer_kwargs):
        self.module = self._load_module()
        if self.config.torch_compile is True:
            self.module = torch.compile(self.module)  # type: ignore
        self.module.hparams.datamodule.batch_size = self.batch_size
        self.module.hparams.datamodule.num_workers = self.config.num_workers

        self.trainer = pl.Trainer(
            logger=False,
            callbacks=[
                hydra.utils.instantiate(
                    self.module.hparams.callbacks.prediction_writer, destination=self.destination, **writer_kwargs
                ),
                hydra.utils.instantiate(self.module.hparams.callbacks.progress_bar),
            ],
            accelerator=self.device_name,
            devices=1,
        )

    def _load_module(self) -> pl.LightningModule:
        raise NotImplementedError

    def delete_module_and_trainer(self) -> None:
        del self.module, self.trainer

    def apply_module(self, input_file: Path) -> None:
        datamodule = self._load_datamodule(input_file)
        assert self.trainer is not None
        self.trainer.predict(model=self.module, dataloaders=[datamodule.predict_dataloader()], return_predictions=False)

    def _load_datamodule(self, input_file: Path) -> DataModule:
        raise NotImplementedError

    def export_prediction(self) -> str:
        raise NotImplementedError


class TypoModuleProcessor(BaseModuleProcessor):
    def _load_module(self) -> pl.LightningModule:
        typer.echo("Loading typo module", err=True)
        checkpoint_path: Path = download_checkpoint(module="typo", model_size=self.model_size)
        return TypoModule.fast_load_from_checkpoint(checkpoint_path, map_location=self.device)

    def _load_datamodule(self, input_file: Path) -> DataModule:
        assert self.module is not None
        self.module.hparams.datamodule.predict.texts = _split_into_documents(input_file.read_text())
        datamodule = DataModule(cfg=self.module.hparams.datamodule)
        datamodule.setup(stage=TrainerFn.PREDICTING)
        return datamodule

    def export_prediction(self) -> str:
        return self.destination.read_text()


class SenterModuleProcessor(BaseModuleProcessor):
    doc_idx = 0

    def load(self):
        if self.model_size != ModelSize.tiny:
            super().load()

    def _load_module(self) -> pl.LightningModule:
        if self.model_size != ModelSize.tiny:
            typer.echo("Loading senter module", err=True)
            checkpoint_path: Path = download_checkpoint(module="senter", model_size=self.model_size)
            return SenterModule.fast_load_from_checkpoint(checkpoint_path, map_location=self.device)
        return  # type: ignore

    def _load_datamodule(self, input_file: Path) -> DataModule:
        assert self.module is not None
        self.module.hparams.datamodule.predict.texts = _split_into_documents(input_file.read_text())
        datamodule = DataModule(cfg=self.module.hparams.datamodule)
        datamodule.setup(stage=TrainerFn.PREDICTING)
        return datamodule

    def apply_module(self, input_file: Path) -> None:
        if self.model_size != ModelSize.tiny:
            super().apply_module(input_file)
            return

        senter = RegexSenter()
        output_string = ""
        doc_id_prefix = datetime.now().strftime("%Y%m%d%H%M")
        for document_text in _split_into_documents(input_file.read_text()):
            document = senter.apply_to_document(document_text)
            document.doc_id = f"{doc_id_prefix}-{self.doc_idx}"
            self.doc_idx += 1
            for sent_idx, sentence in enumerate(document.sentences):
                sentence.sid = f"{document.doc_id}-{sent_idx}"
                sentence.misc_comment = f"kwja:{kwja.__version__}"
            output_string += document.to_raw_text()
        self.destination.write_text(output_string)

    def export_prediction(self) -> str:
        return self.destination.read_text()


class Seq2SeqModuleProcessor(BaseModuleProcessor):
    def _load_module(self):
        typer.echo("Loading seq2seq module", err=True)
        checkpoint_path: Path = download_checkpoint(module="seq2seq", model_size=self.model_size)
        return Seq2SeqModule.fast_load_from_checkpoint(checkpoint_path, map_location=self.device)

    def _load_datamodule(self, input_file: Path) -> DataModule:
        assert self.module is not None
        self.module.hparams.datamodule.predict.senter_file = input_file
        datamodule = DataModule(cfg=self.module.hparams.datamodule)
        datamodule.setup(stage=TrainerFn.PREDICTING)
        return datamodule

    def export_prediction(self) -> str:
        return self.destination.read_text()


class CharModuleProcessor(BaseModuleProcessor):
    def _load_module(self) -> pl.LightningModule:
        typer.echo("Loading char module", err=True)
        checkpoint_path: Path = download_checkpoint(module="char", model_size=self.model_size)
        return CharModule.fast_load_from_checkpoint(checkpoint_path, map_location=self.device)

    def _load_datamodule(self, input_file: Path) -> DataModule:
        assert self.module is not None
        self.module.hparams.datamodule.predict.senter_file = input_file
        datamodule = DataModule(cfg=self.module.hparams.datamodule)
        datamodule.setup(stage=TrainerFn.PREDICTING)
        return datamodule

    def export_prediction(self) -> str:
        export_text = ""
        with self.destination.open() as f:
            for juman_text in chunk_by_sentence(f):
                sentence = Sentence.from_jumanpp(juman_text)
                if sentence.comment != "":
                    export_text += sentence.comment + "\n"
                export_text += " ".join(m.text for m in sentence.morphemes) + "\n"
        return export_text


class WordModuleProcessor(BaseModuleProcessor):
    def __init__(self, config: CLIConfig, batch_size: int, from_seq2seq: bool) -> None:
        super().__init__(config, batch_size)
        self.from_seq2seq = from_seq2seq

    def load(self):
        super().load(preserve_reading_lemma_canon=self.from_seq2seq)

    def _load_module(self) -> pl.LightningModule:
        typer.echo("Loading word module", err=True)
        checkpoint_path: Path = download_checkpoint(module="word", model_size=self.model_size)
        return WordModule.fast_load_from_checkpoint(checkpoint_path, map_location=self.device)

    def _load_datamodule(self, input_file: Path) -> DataModule:
        assert self.module is not None
        self.module.hparams.datamodule.predict.juman_file = input_file
        datamodule = DataModule(cfg=self.module.hparams.datamodule)
        datamodule.setup(stage=TrainerFn.PREDICTING)
        return datamodule

    def export_prediction(self) -> str:
        return self.destination.read_text()


class CLIProcessor:
    def __init__(self, config: CLIConfig, tasks: List[str]) -> None:
        self.raw_destination = Path(NamedTemporaryFile(suffix=".txt", delete=False).name)
        self._task2processors: Dict[str, BaseModuleProcessor] = {
            "typo": TypoModuleProcessor(config, config.typo_batch_size),
            "senter": SenterModuleProcessor(config, config.senter_batch_size),
            "seq2seq": Seq2SeqModuleProcessor(config, config.seq2seq_batch_size),
            "char": CharModuleProcessor(config, config.char_batch_size),
            "word": WordModuleProcessor(
                config,
                config.word_batch_size,
                from_seq2seq="seq2seq" in tasks,
            ),
        }
        self.processors: List[BaseModuleProcessor] = [self._task2processors[task] for task in tasks]

    def load_all_modules(self) -> None:
        for processor in self.processors:
            processor.load()

    def refresh(self) -> None:
        self.raw_destination.unlink(missing_ok=True)
        for processor in self._task2processors.values():
            processor.destination.unlink(missing_ok=True)

    def run(self, input_documents: List[str], interactive: bool = False) -> None:
        self.raw_destination.write_text(
            "".join(_normalize_text(input_document) + "\nEOD\n" for input_document in input_documents)
        )
        input_file = self.raw_destination
        for processor in self.processors:
            if interactive is False:
                processor.load()
            processor.apply_module(input_file)
            input_file = processor.destination
            if interactive is False:
                processor.delete_module_and_trainer()
        print(self.processors[-1].export_prediction(), end="")


def _normalize_text(text: str) -> str:
    # Tokenizers (BertJapaneseTokenizer, DebertaV2Tokenizer, etc.) apply NFKC normalization internally, so
    # there may be inconsistency in number of characters if not applying NFKC normalization in advance
    normalized = normalize("NFKC", text)
    # escape several symbols and delete control characters
    normalized = normalized.translate(TRANSLATION_TABLE)
    # prevent hydra.utils.instantiate from interpolating the string "${...}"
    normalized = OMEGACONF_VARIABLE_INTERPOLATION.sub(r"$␣\g<variable>", normalized)
    # if normalized != text:
    #     typer.echo(f"apply normalization ({text} -> {normalized})", err=True)
    return normalized


def _split_into_documents(input_text: str) -> List[str]:
    documents: List[str] = []
    document: str = ""
    for line in input_text.split("\n"):
        if line == "EOD":
            documents.append(document.rstrip())
            document = ""
        else:
            document += f"{line}\n"
    else:
        if document.rstrip() != "":
            documents.append(document.rstrip())
    return documents


def version_callback(value: bool) -> None:
    if value is True:
        typer.echo(f"KWJA {kwja.__version__}")
        raise typer.Exit()


def tasks_callback(value: str) -> str:
    """sort and validate specified tasks"""
    values: List[str] = [v for v in value.split(",") if v]
    tasks: List[str] = []
    for candidate_task in ("typo", "senter", "seq2seq", "char", "word"):
        if candidate_task in values:
            tasks.append(candidate_task)
            values.remove(candidate_task)
    if len(values) == 1:
        raise typer.BadParameter(f"invalid task is specified: {repr(values[0])}")
    if len(values) > 1:
        raise typer.BadParameter(f"invalid tasks are specified: {', '.join(repr(v) for v in values)}")
    if len(tasks) == 0:
        raise typer.BadParameter("task must be specified")
    valid_task_combinations: Set[Tuple[str, ...]] = {
        ("typo",),
        ("typo", "senter"),
        ("typo", "senter", "char"),
        ("typo", "senter", "char", "word"),
        ("typo", "senter", "seq2seq"),
        ("typo", "senter", "seq2seq", "word"),
        ("senter",),
        ("senter", "char"),
        ("senter", "char", "word"),
        ("senter", "seq2seq"),
        ("senter", "seq2seq", "word"),
    }

    if tuple(tasks) not in valid_task_combinations:
        raise typer.BadParameter(
            "task combination is invalid. "
            f"Please specify one of {', '.join(repr(','.join(ts)) for ts in valid_task_combinations)}."
        )
    return ",".join(tasks)


@app.command()
def main(
    text: Optional[str] = typer.Option(None, help="Text to be analyzed."),
    filename: List[Path] = typer.Option([], dir_okay=False, help="Files to be analyzed."),
    model_size: Optional[ModelSize] = typer.Option(None, help="Model size to be used."),
    device: Optional[Device] = typer.Option(None, help="Device to be used."),
    typo_batch_size: Optional[int] = typer.Option(None, help="Batch size for typo module."),
    senter_batch_size: Optional[int] = typer.Option(None, help="Batch size for senter module."),
    seq2seq_batch_size: Optional[int] = typer.Option(None, help="Batch size for seq2seq module."),
    char_batch_size: Optional[int] = typer.Option(None, help="Batch size for char module."),
    word_batch_size: Optional[int] = typer.Option(None, help="Batch size for word module."),
    tasks: str = typer.Option("senter,char,word", callback=tasks_callback, help="Tasks to be performed."),
    _: Optional[bool] = typer.Option(None, "--version", callback=version_callback, is_eager=True),
    config_file: Optional[Path] = typer.Option(None, help="Path to KWJA config file."),
) -> None:
    input_text: Optional[str] = None
    if text is not None and len(filename) > 0:
        typer.echo("ERROR: Please provide text or filename, not both", err=True)
        raise typer.Abort()
    elif text is not None:
        input_text = text
    elif len(filename) > 0:
        input_text = "".join(path.read_text().rstrip("\n") + "\nEOD\n" for path in filename)
    elif sys.stdin.isatty() is False:
        input_text = sys.stdin.read()
    else:
        pass  # interactive mode

    specified_tasks: List[str] = tasks.split(",")

    if config_file is None:
        config_file = get_kwja_config_file()
    if config_file.exists():
        config = CLIConfig.from_yaml(config_file)
    else:
        config = CLIConfig()
    if model_size is not None:
        config.model_size = model_size
    if device is not None:
        config.device = device
    if typo_batch_size is not None:
        config.typo_batch_size = typo_batch_size
    if senter_batch_size is not None:
        config.senter_batch_size = senter_batch_size
    if seq2seq_batch_size is not None:
        config.seq2seq_batch_size = seq2seq_batch_size
    if char_batch_size is not None:
        config.char_batch_size = char_batch_size
    if word_batch_size is not None:
        config.word_batch_size = word_batch_size

    processor = CLIProcessor(config, specified_tasks)

    # Batch mode
    if input_text is not None:
        if input_text.strip() != "":
            processor.run(_split_into_documents(input_text))
        processor.refresh()
        raise typer.Exit()

    # Interactive mode
    processor.load_all_modules()
    typer.echo('Please end your input with a new line and type "EOD"', err=True)
    input_text = ""
    while True:
        input_ = input()
        if input_ == "EOD":
            processor.refresh()
            processor.run([input_text], interactive=True)
            print("EOD")  # To indicate the end of the output.
            input_text = ""
        else:
            input_text += input_ + "\n"


if __name__ == "__main__":
    typer.run(main)
