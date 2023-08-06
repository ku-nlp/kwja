import sys
from io import TextIOBase
from pathlib import Path
from typing import Any, Optional, Sequence, TextIO, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter

import kwja
from kwja.callbacks.utils import convert_char_predictions_into_tags, set_morphemes, set_sentences
from kwja.datamodule.datasets import CharDataset, CharInferenceDataset
from kwja.datamodule.examples import CharExample, CharInferenceExample
from kwja.utils.sub_document import extract_target_sentences, to_orig_doc_id


class CharModuleWriter(BasePredictionWriter):
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
        dataset: Union[CharDataset, CharInferenceDataset] = dataloaders[dataloader_idx].dataset

        special_ids = set(dataset.tokenizer.all_special_ids) - {dataset.tokenizer.unk_token_id}
        for example_id, sent_segmentation_predictions, word_segmentation_predictions, word_norm_op_predictions in zip(
            prediction["example_ids"].tolist(),
            prediction["sent_segmentation_predictions"].tolist(),
            prediction["word_segmentation_predictions"].tolist(),
            prediction["word_norm_op_predictions"].tolist(),
        ):
            example: Union[CharExample, CharInferenceExample] = dataset.examples[example_id]
            assert example.doc_id is not None, "doc_id isn't set"
            document = dataset.doc_id2document.pop(example.doc_id)

            sent_segmentation_tags, word_segmentation_tags, word_norm_op_tags = convert_char_predictions_into_tags(
                sent_segmentation_predictions,
                word_segmentation_predictions,
                word_norm_op_predictions,
                example.encoding.input_ids,
                special_ids,
            )
            set_sentences(document, sent_segmentation_tags)
            set_morphemes(document, word_segmentation_tags, word_norm_op_tags)

            orig_doc_id = to_orig_doc_id(document.doc_id)
            if orig_doc_id != self.prev_doc_id:
                self.prev_doc_id = orig_doc_id
                self.prev_sid = 1  # 1-origin
            sentences = extract_target_sentences(document)

            output_string = "".join(
                f"# S-ID:{orig_doc_id}-{self.prev_sid + i} kwja:{kwja.__version__}\n{s.to_jumanpp()}"
                for i, s in enumerate(sentences)
            )
            if isinstance(self.destination, Path):
                with self.destination.open(mode="a") as f:
                    f.write(output_string)
            elif isinstance(self.destination, TextIOBase):
                self.destination.write(output_string)

            self.prev_sid += len(sentences)

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]] = None,
    ) -> None:
        pass  # pragma: no cover
