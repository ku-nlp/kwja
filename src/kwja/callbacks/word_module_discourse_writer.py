import logging
import os
import sys
from io import TextIOBase
from pathlib import Path
from typing import Any, Optional, Sequence, TextIO, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from rhoknp import Document, Sentence

from kwja.datamodule.datasets import WordDataset, WordInferenceDataset
from kwja.datamodule.datasets.word_dataset import WordExampleSet
from kwja.datamodule.datasets.word_inference_dataset import WordInferenceExample
from kwja.utils.constants import INDEX2DISCOURSE_RELATION
from kwja.utils.sub_document import extract_target_sentences

logger = logging.getLogger(__name__)


class WordModuleDiscourseWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        pred_filename: str = "predict",
        use_stdout: bool = False,
    ) -> None:
        super().__init__(write_interval="epoch")

        self.destination: Union[Path, TextIO]
        if use_stdout is True:
            self.destination = sys.stdout
        else:
            self.destination = Path(f"{output_dir}/{pred_filename}.knp")
            self.destination.parent.mkdir(exist_ok=True, parents=True)
            if self.destination.exists():
                os.remove(str(self.destination))

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]] = None,
    ) -> None:
        sentences: list[Sentence] = []
        dataloaders = trainer.predict_dataloaders
        for prediction in predictions:
            for batch_pred in prediction:
                batch_tokens = batch_pred["tokens"]
                batch_example_ids = batch_pred["example_ids"]
                dataloader_idx: int = batch_pred["dataloader_idx"]
                dataset: Union[WordDataset, WordInferenceDataset] = dataloaders[dataloader_idx].dataset
                batch_discourse_parsing_preds = torch.argmax(batch_pred["discourse_parsing_logits"], dim=3)
                for (tokens, example_id, discourse_parsing_preds,) in zip(
                    batch_tokens,
                    batch_example_ids,
                    batch_discourse_parsing_preds.tolist(),
                ):
                    example: Union[WordExampleSet, WordInferenceExample] = dataset.examples[example_id]
                    doc_id = example.doc_id
                    document = dataset.doc_id2document[doc_id]
                    document.doc_id = doc_id
                    document = document.reparse()  # reparse to get clauses
                    self._add_discourse(document, discourse_parsing_preds)
                    sentences += extract_target_sentences(document)

        output_string = "".join(sentence.to_knp() for sentence in sentences)
        if isinstance(self.destination, Path):
            self.destination.write_text(output_string)
        elif isinstance(self.destination, TextIOBase):
            self.destination.write(output_string)

    @staticmethod
    def _add_discourse(document: Document, discourse_preds: list[list[int]]) -> None:
        if document.need_clause_tag:
            logger.warning("failed to output clause boundaries")
            return

        for modifier in document.clauses:
            modifier_morpheme_index = modifier.end.morphemes[0].global_index
            preds = []
            if "談話関係" in modifier.end.features:
                del modifier.end.features["談話関係"]
            for head in document.clauses:
                if modifier == head:
                    continue
                head_sid = head.sentence.sid
                head_morpheme_index = head.end.morphemes[0].global_index
                head_base_phrase_index = head.end.index
                pred = INDEX2DISCOURSE_RELATION[discourse_preds[modifier_morpheme_index][head_morpheme_index]]
                if pred != "談話関係なし":
                    preds.append(f"{head_sid}/{head_base_phrase_index}/{pred}")
            if preds:
                modifier.end.features["談話関係"] = ";".join(preds)

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
        pass
