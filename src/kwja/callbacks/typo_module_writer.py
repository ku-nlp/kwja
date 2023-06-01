import sys
from io import TextIOBase
from pathlib import Path
from typing import Any, Optional, Sequence, TextIO, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
from transformers import PreTrainedTokenizerBase

from kwja.callbacks.utils import apply_edit_operations, convert_predictions_into_typo_corr_op_tags, get_maps
from kwja.datamodule.datasets import TypoDataset, TypoInferenceDataset
from kwja.datamodule.examples import TypoExample, TypoInferenceExample
from kwja.utils.constants import RESOURCE_PATH


class TypoModuleWriter(BasePredictionWriter):
    def __init__(
        self,
        confidence_threshold: float,
        tokenizer: PreTrainedTokenizerBase,
        destination: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__(write_interval="batch")
        if destination is None:
            self.destination: Union[Path, TextIO] = sys.stdout
        else:
            if isinstance(destination, str):
                destination = Path(destination)
            self.destination = destination
            self.destination.parent.mkdir(exist_ok=True, parents=True)
            self.destination.unlink(missing_ok=True)

        self.confidence_threshold = confidence_threshold

        self.token2token_id, self.token_id2token = get_maps(
            tokenizer,
            RESOURCE_PATH / "typo_correction" / "multi_char_vocab.txt",
        )

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
        dataset: Union[TypoDataset, TypoInferenceDataset] = dataloaders[dataloader_idx].dataset

        post_texts = []
        for example_id, kdr_predictions, kdr_probabilities, ins_predictions, ins_probabilities in zip(
            prediction["example_ids"].tolist(),
            prediction["kdr_predictions"].tolist(),
            prediction["kdr_probabilities"].tolist(),
            prediction["ins_predictions"].tolist(),
            prediction["ins_probabilities"].tolist(),
        ):
            if example_id in dataset.stash:
                texts = dataset.stash.pop(example_id)
                post_texts.extend(texts)

            example: Union[TypoExample, TypoInferenceExample] = dataset.examples[example_id]
            seq_len: int = len(example.pre_text)
            if seq_len == 0:
                continue

            args = (self.confidence_threshold, self.token2token_id, self.token_id2token)
            kdr_tags = convert_predictions_into_typo_corr_op_tags(kdr_predictions, kdr_probabilities, "R", *args)
            ins_tags = convert_predictions_into_typo_corr_op_tags(ins_predictions, ins_probabilities, "I", *args)

            # the prediction of the first token (= [CLS]) is excluded.
            # the prediction of the dummy token at the end is used for insertion only.
            post_text = apply_edit_operations(example.pre_text, kdr_tags[1 : seq_len + 1], ins_tags[1 : seq_len + 2])
            post_texts.append(post_text)

        if batch_idx == len(dataloaders[dataloader_idx]) - 1:
            for texts in dataset.stash.values():
                post_texts.extend(texts)
            dataset.stash.clear()

        output_string = "".join(post_text + "\nEOD\n" for post_text in post_texts)
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
