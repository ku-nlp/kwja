import logging
import os
import re
import sys
from abc import ABC
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Iterator, List, Optional, Set, TextIO, Tuple
from unicodedata import normalize

import hydra
import pytorch_lightning as pl
import torch
import typer
from pytorch_lightning.trainer.states import TrainerFn
from rhoknp import Document, Sentence
from rhoknp.utils.reader import chunk_by_document, chunk_by_sentence
from typing_extensions import Annotated

import kwja
from kwja.cli.config import CLIConfig, Device, ModelSize, get_kwja_config_file
from kwja.cli.utils import download_checkpoint, prepare_device
from kwja.datamodule.datamodule import DataModule
from kwja.datamodule.datasets.utils import add_doc_ids, add_sent_ids
from kwja.modules import CharModule, Seq2SeqModule, TypoModule, WordModule
from kwja.utils.constants import TRANSLATION_TABLE
from kwja.utils.logging_util import filter_logs

filter_logs(environment="production")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
OMEGACONF_VARIABLE_INTERPOLATION = re.compile(r"\$(?P<variable>\{.*?})")
logging.basicConfig(format="")

logger = logging.getLogger("kwja_cli")
logger.setLevel(logging.INFO)
app = typer.Typer(pretty_exceptions_show_locals=False)


class InputFormat(str, Enum):
    RAW = "raw"
    JUMANPP = "jumanpp"
    KNP = "knp"


class BaseModuleProcessor(ABC):
    input_format: InputFormat

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
    input_format = InputFormat.RAW

    def _load_module(self) -> pl.LightningModule:
        logger.info("Loading typo module")
        checkpoint_path: Path = download_checkpoint(module="typo", model_size=self.model_size)
        return TypoModule.fast_load_from_checkpoint(checkpoint_path, map_location=self.device)

    def _load_datamodule(self, input_file: Path) -> DataModule:
        assert self.module is not None
        with input_file.open() as f:
            self.module.hparams.datamodule.predict.texts = list(_chunk_by_document(f, self.input_format))
        datamodule = DataModule(cfg=self.module.hparams.datamodule)
        datamodule.setup(stage=TrainerFn.PREDICTING)
        return datamodule

    def export_prediction(self) -> str:
        output_text = ""
        for line in self.destination.read_text().strip().split("\n"):
            if line.startswith("# D-ID:"):
                pass
            else:
                output_text += line + "\n"
        return output_text


class CharModuleProcessor(BaseModuleProcessor):
    input_format = InputFormat.RAW

    def _load_module(self) -> pl.LightningModule:
        logger.info("Loading char module")
        checkpoint_path: Path = download_checkpoint(module="char", model_size=self.model_size)
        return CharModule.fast_load_from_checkpoint(checkpoint_path, map_location=self.device)

    def _load_datamodule(self, input_file: Path) -> DataModule:
        assert self.module is not None
        with input_file.open() as f:
            self.module.hparams.datamodule.predict.texts = list(_chunk_by_document(f, self.input_format))
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


class Seq2SeqModuleProcessor(BaseModuleProcessor):
    input_format = InputFormat.JUMANPP

    def _load_module(self):
        logger.info("Loading seq2seq module")
        checkpoint_path: Path = download_checkpoint(module="seq2seq", model_size=self.model_size)
        return Seq2SeqModule.fast_load_from_checkpoint(checkpoint_path, map_location=self.device)

    def _load_datamodule(self, input_file: Path) -> DataModule:
        assert self.module is not None
        self.module.hparams.datamodule.predict.juman_file = input_file
        datamodule = DataModule(cfg=self.module.hparams.datamodule)
        datamodule.setup(stage=TrainerFn.PREDICTING)
        return datamodule

    def export_prediction(self) -> str:
        return self.destination.read_text()


class WordModuleProcessor(BaseModuleProcessor):
    input_format = InputFormat.JUMANPP

    def __init__(self, config: CLIConfig, batch_size: int, from_seq2seq: bool) -> None:
        super().__init__(config, batch_size)
        self.from_seq2seq = from_seq2seq

    def load(self):
        super().load(preserve_reading_lemma_canon=self.from_seq2seq)

    def _load_module(self) -> pl.LightningModule:
        logger.info("Loading word module")
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
        self.initial_destination = Path(NamedTemporaryFile(delete=False).name)
        self._task2processors: Dict[str, BaseModuleProcessor] = {
            "typo": TypoModuleProcessor(config, config.typo_batch_size),
            "char": CharModuleProcessor(config, config.char_batch_size),
            "seq2seq": Seq2SeqModuleProcessor(config, config.seq2seq_batch_size),
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
        self.initial_destination.unlink(missing_ok=True)
        for processor in self._task2processors.values():
            processor.destination.unlink(missing_ok=True)

    def run(self, input_documents: List[Document], interactive: bool = False) -> str:
        input_documents = [document for document in input_documents if document.text != ""]
        if len(input_documents) == 0:
            return ""
        add_doc_ids(input_documents)
        add_sent_ids(input_documents)
        if self.processors[0].input_format == InputFormat.RAW:
            output_text = ""
            for input_document in input_documents:
                output_text += f"# D-ID:{input_document.doc_id}\n"
                output_text += normalize_text(input_document.text) + "\nEOD\n"
            self.initial_destination.write_text(output_text)
        elif self.processors[0].input_format == InputFormat.JUMANPP:
            self.initial_destination.write_text("".join(document.to_jumanpp() + "\n" for document in input_documents))
        else:
            raise AssertionError  # unreachable

        input_file = self.initial_destination
        for processor in self.processors:
            if processor.module is None or processor.trainer is None:
                processor.load()
            processor.apply_module(input_file)
            input_file = processor.destination
            if interactive is False:
                processor.delete_module_and_trainer()
        return self.processors[-1].export_prediction()


def normalize_text(text: str) -> str:
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


def _chunk_by_document(f: TextIO, input_format: InputFormat) -> Iterator[str]:
    if input_format in (InputFormat.JUMANPP, InputFormat.KNP):
        yield from chunk_by_document(f)
    elif input_format == InputFormat.RAW:
        buff: str = ""
        for line in f:
            if line.strip() == "EOD":
                yield buff.rstrip()
                buff = ""
            else:
                buff += line
        if buff.rstrip() != "":
            yield buff.rstrip()
    else:
        raise AssertionError  # unreachable


def _load_document_from_text(text: str, input_format: InputFormat) -> Document:
    if input_format == InputFormat.RAW:
        return Document.from_raw_text(text)
    elif input_format == InputFormat.JUMANPP:
        return Document.from_jumanpp(text)
    elif input_format == InputFormat.KNP:
        return Document.from_knp(text)
    else:
        raise AssertionError  # unreachable


def _version_callback(value: bool) -> None:
    if value is True:
        print(f"KWJA {kwja.__version__}")
        raise typer.Exit


def _tasks_callback(value: str) -> str:
    """sort and validate specified tasks"""
    values: List[str] = [v for v in value.split(",") if v]
    tasks: List[str] = []
    for candidate_task in ("typo", "char", "seq2seq", "word"):
        if candidate_task in values:
            tasks.append(candidate_task)
            values.remove(candidate_task)
    if len(values) == 1:
        raise typer.BadParameter(f"invalid task is specified: {values[0]!r}")
    if len(values) > 1:
        raise typer.BadParameter(f"invalid tasks are specified: {', '.join(repr(v) for v in values)}")
    if len(tasks) == 0:
        raise typer.BadParameter("task must be specified")
    return ",".join(tasks)


@app.command()
def main(
    text: Annotated[Optional[str], typer.Option(help="Text to be analyzed.")] = None,
    filename: List[Path] = typer.Option([], dir_okay=False, help="Files to be analyzed."),
    model_size: Annotated[
        Optional[ModelSize], typer.Option(case_sensitive=False, help="Model size to be used.")
    ] = None,
    device: Annotated[Optional[Device], typer.Option(case_sensitive=False, help="Device to be used.")] = None,
    typo_batch_size: Annotated[Optional[int], typer.Option(help="Batch size for typo module.")] = None,
    char_batch_size: Annotated[Optional[int], typer.Option(help="Batch size for char module.")] = None,
    seq2seq_batch_size: Annotated[Optional[int], typer.Option(help="Batch size for seq2seq module.")] = None,
    word_batch_size: Annotated[Optional[int], typer.Option(help="Batch size for word module.")] = None,
    tasks: Annotated[str, typer.Option(callback=_tasks_callback, help="Tasks to be performed.")] = "char,word",
    _: Annotated[
        Optional[bool],
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version and exit."),
    ] = None,
    config_file: Annotated[Optional[Path], typer.Option(help="Path to KWJA config file.")] = None,
    input_format: Annotated[InputFormat, typer.Option(case_sensitive=False, help="Input format.")] = InputFormat.RAW,
) -> None:
    # validate task combination
    specified_tasks: List[str] = tasks.split(",")
    valid_task_combinations: Set[Tuple[str, ...]] = {
        ("typo",),
        ("typo", "char"),
        ("typo", "char", "seq2seq"),
        ("typo", "char", "word"),
        ("typo", "char", "seq2seq", "word"),
        ("char",),
        ("char", "seq2seq"),
        ("char", "word"),
        ("char", "seq2seq", "word"),
    }
    if input_format in (InputFormat.JUMANPP, InputFormat.KNP):
        valid_task_combinations |= {
            ("seq2seq",),
            ("seq2seq", "word"),
            ("word",),
        }
    if tuple(specified_tasks) not in valid_task_combinations:
        raise typer.BadParameter(
            "task combination is invalid. "
            f"Please specify one of {', '.join(repr(','.join(ts)) for ts in valid_task_combinations)}."
        )
    if input_format in (InputFormat.JUMANPP, InputFormat.KNP):
        if specified_tasks[0] in ("typo", "char"):
            logger.warning("WARNING: with typo or char task, your input text will be treated as raw text.")
        elif specified_tasks[0] in ("seq2seq", "word"):
            logger.warning("WARNING: with seq2seq or word task, your input text will be treated as a word sequence.")

    input_documents: Optional[List[Document]] = None
    if text is not None and len(filename) > 0:
        logger.error("ERROR: Please provide text or filename, not both")
        raise typer.Abort
    elif text is not None:
        input_documents = [_load_document_from_text(text, input_format)]
    elif len(filename) > 0:
        input_documents = []
        for path in filename:
            if path.exists() is False:
                logger.error(f"ERROR: {path} does not exist")
                raise typer.Abort
            with path.open() as f:
                for document_text in _chunk_by_document(f, input_format):
                    input_documents.append(_load_document_from_text(document_text, input_format))
    else:
        pass  # interactive mode

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
    if char_batch_size is not None:
        config.char_batch_size = char_batch_size
    if seq2seq_batch_size is not None:
        config.seq2seq_batch_size = seq2seq_batch_size
    if word_batch_size is not None:
        config.word_batch_size = word_batch_size

    processor = CLIProcessor(config, specified_tasks)

    # Batch mode
    if input_documents is not None:
        output: str = processor.run(input_documents)
        print(output or "EOD\n", end="")
        processor.refresh()
        raise typer.Exit

    # Interactive mode
    processor.load_all_modules()
    print('Type "EOD" in a new line to finish the input.', file=sys.stderr)
    input_text = ""
    while True:
        try:
            input_ = input()
        except EOFError:
            break
        if input_ == "EOD":
            processor.refresh()
            input_document: Document = _load_document_from_text(input_text, input_format)
            output = processor.run([input_document], interactive=True)
            print(output, end="")
            if specified_tasks != ["typo"] or output == "":
                print("EOD")  # To indicate the end of the output.
            input_text = ""
        else:
            input_text += input_ + "\n"


if __name__ == "__main__":
    typer.run(main)
