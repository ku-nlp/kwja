import sys
from io import TextIOBase
from pathlib import Path
from typing import Any, Optional, Sequence, TextIO, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter

import kwja
from kwja.callbacks.utils import convert_senter_predictions_into_tags
from kwja.datamodule.datasets.senter import SenterDataset
from kwja.datamodule.datasets.senter_inference import SenterInferenceDataset, SenterInferenceExample
from kwja.datamodule.examples import SenterExample
from kwja.utils.sub_document import to_orig_doc_id


class SenterModuleWriter(BasePredictionWriter):
    def __init__(self, destination: Optional[Union[str, Path]] = None) -> None:
        super().__init__(write_interval="batch")
        if destination is None:
            self.destination: Union[Path, TextIO] = sys.stdout
        else:
            if isinstance(destination, str):
                destination = Path(destination)
            self.destination = destination
            self.destination.parent.mkdir(exist_ok=True, parents=True)
            self.destination.unlink(missing_ok=True)

        self.prev_doc_id: Optional[str] = None
        self.prev_sid: int = 0

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
        if isinstance(trainer.predict_dataloaders, dict):
            dataloaders = list(trainer.predict_dataloaders.values())
        dataset: Union[SenterDataset, SenterInferenceDataset] = dataloaders[dataloader_idx].dataset

        special_ids = set(dataset.tokenizer.all_special_ids) - {dataset.tokenizer.unk_token_id}

        output_string: str = ""
        for example_id, sent_segmentation_predictions in zip(
            prediction["example_ids"].tolist(),
            prediction["sent_segmentation_predictions"].tolist(),
        ):
            example: Union[SenterExample, SenterInferenceExample] = dataset.examples[example_id]
            assert example.doc_id is not None, "doc_id isn't set"
            document = dataset.doc_id2document.pop(example.doc_id)

            sent_segmentation_tags = convert_senter_predictions_into_tags(
                sent_segmentation_predictions, example.encoding.input_ids, special_ids
            )

            orig_doc_id = to_orig_doc_id(document.doc_id)

            is_new_document = orig_doc_id != self.prev_doc_id
            is_first_document = self.prev_doc_id is None
            if is_new_document:
                self.prev_doc_id = orig_doc_id
                self.prev_sid = 0
                if not is_first_document:
                    output_string += "\n"

            for char_index, (char, sent_segmentation_tag) in enumerate(zip(document.text, sent_segmentation_tags)):
                if is_new_document and char_index == 0:
                    # The first character of the document is always the start of a sentence
                    output_string += f"# S-ID:{orig_doc_id}-{self.prev_sid + 1} kwja:{kwja.__version__}\n"
                    self.prev_sid += 1
                elif sent_segmentation_tag == "B":
                    output_string += "\n"
                    output_string += f"# S-ID:{orig_doc_id}-{self.prev_sid + 1} kwja:{kwja.__version__}\n"
                    self.prev_sid += 1
                output_string += char

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

    def on_predict_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        output_string: str = "\n"
        if isinstance(self.destination, Path):
            with self.destination.open(mode="a") as f:
                f.write(output_string)
        elif isinstance(self.destination, TextIOBase):
            self.destination.write(output_string)
