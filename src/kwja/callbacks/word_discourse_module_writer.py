import logging
import os
import sys
from io import TextIOBase
from pathlib import Path
from typing import Any, Optional, Sequence, TextIO, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter

from kwja.datamodule.datasets import WordDataset, WordInferenceDataset
from kwja.datamodule.datasets.word_dataset import WordExampleSet
from kwja.datamodule.datasets.word_inference_dataset import WordInferenceExample
from kwja.utils.sub_document import extract_target_sentences
from kwja.utils.word_module_writer import add_discourse

logger = logging.getLogger(__name__)


class WordDiscourseModuleWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        use_stdout: bool = False,
        output_filename: str = "predict",
    ) -> None:
        super().__init__(write_interval="batch")
        if use_stdout:
            self.destination: Union[Path, TextIO] = sys.stdout
        else:
            self.destination = Path(output_dir) / f"{output_filename}.knp"
            self.destination.parent.mkdir(exist_ok=True, parents=True)
            if self.destination.exists():
                os.remove(str(self.destination))

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        dataloaders = trainer.predict_dataloaders
        dataset: Union[WordDataset, WordInferenceDataset] = dataloaders[dataloader_idx].dataset

        for example_id, discourse_predictions in zip(
            prediction["example_ids"], prediction["discourse_predictions"].tolist()
        ):
            example: Union[WordExampleSet, WordInferenceExample] = dataset.examples[example_id]
            document = dataset.doc_id2document[example.doc_id]
            # メモリリーク対策
            predicted_document = document.reparse()
            predicted_document.doc_id = document.doc_id
            for predicted_sentence, sentence in zip(predicted_document.sentences, document.sentences):
                predicted_sentence.comment = sentence.comment

            add_discourse(predicted_document, discourse_predictions)

            output_string = "".join(s.to_knp() for s in extract_target_sentences(predicted_document))
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
        pass
