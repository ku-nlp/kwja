import sys
from io import TextIOBase
from pathlib import Path
from typing import Any, List, Optional, Sequence, TextIO, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
from transformers import PreTrainedTokenizerBase

from kwja.datamodule.datasets import Seq2SeqDataset, Seq2SeqInferenceDataset
from kwja.datamodule.examples import Seq2SeqExample, Seq2SeqInferenceExample
from kwja.utils.constants import NEW_LINE_TOKEN


class Seq2SeqModuleWriter(BasePredictionWriter):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, destination: Optional[Union[str, Path]] = None) -> None:
        super().__init__(write_interval="batch")
        if destination is None:
            self.destination: Union[Path, TextIO] = sys.stdout
        else:
            if isinstance(destination, str):
                destination = Path(destination)
            self.destination = destination
            self.destination.parent.mkdir(exist_ok=True, parents=True)
            self.destination.unlink(missing_ok=True)

        self.tokenizer: PreTrainedTokenizerBase = tokenizer

    @staticmethod
    def _shape(input_text: str) -> str:
        output_lines: List[str] = []
        if input_text.startswith("</s>"):
            refined_input_text: str = input_text[len("</s>") :]
        else:
            refined_input_text = input_text
        for line in refined_input_text.replace(NEW_LINE_TOKEN, "\n").split("\n"):
            output_lines.append(line.lstrip().rstrip())
        return "\n".join(output_lines)

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        dataloaders = trainer.predict_dataloaders
        dataset: Union[Seq2SeqDataset, Seq2SeqInferenceDataset] = dataloaders[dataloader_idx].dataset

        outputs: List[str] = []
        for example_id, seq2seq_predictions in zip(
            prediction["example_ids"].tolist(), prediction["seq2seq_predictions"].tolist()
        ):
            example: Union[Seq2SeqExample, Seq2SeqInferenceExample] = dataset.examples[example_id]
            seq_len: int = len(example.src_text)
            if seq_len == 0:
                continue

            decoded: str = self.tokenizer.decode(
                [x for x in seq2seq_predictions if x != self.tokenizer.pad_token_id], skip_special_tokens=False
            )
            # outputs.append(f"src{dataloader_idx}: {example.src_text}\n")
            outputs.append(self._shape(decoded))
        output_string: str = "".join(outputs)
        if isinstance(self.destination, Path):
            with self.destination.open(mode="a") as f:
                f.write(output_string)
        elif isinstance(self.destination, TextIOBase):
            self.destination.write(output_string)

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]] = None,
    ) -> None:
        pass  # pragma: no cover
