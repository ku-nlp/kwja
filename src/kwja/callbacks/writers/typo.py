from pathlib import Path
from typing import Any, Optional, Sequence, Union

import pytorch_lightning as pl
from transformers import PreTrainedTokenizerBase

from kwja.callbacks.writers.base import BaseModuleWriter
from kwja.callbacks.writers.utils import apply_edit_operations, convert_typo_predictions_into_tags
from kwja.datamodule.datasets import TypoDataset, TypoInferenceDataset
from kwja.datamodule.datasets.typo import get_maps
from kwja.datamodule.examples import TypoExample, TypoInferenceExample
from kwja.utils.constants import RESOURCE_PATH


class TypoModuleWriter(BaseModuleWriter):
    def __init__(
        self,
        confidence_threshold: float,
        tokenizer: PreTrainedTokenizerBase,
        destination: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__(destination=destination)
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
        if isinstance(trainer.predict_dataloaders, dict):
            dataloader = list(trainer.predict_dataloaders.values())[dataloader_idx]
        else:
            dataloader = trainer.predict_dataloaders[dataloader_idx]
        dataset: Union[TypoDataset, TypoInferenceDataset] = dataloader.dataset

        post_texts = []
        for example_id, kdr_predictions, kdr_probabilities, ins_predictions, ins_probabilities in zip(
            *[v.tolist() for v in prediction.values()]
        ):
            if example_id in dataset.stash:
                texts = dataset.stash.pop(example_id)
                post_texts.extend(texts)

            example: Union[TypoExample, TypoInferenceExample] = dataset.examples[example_id]
            seq_len: int = len(example.pre_text)
            if seq_len == 0:
                continue

            args = (self.confidence_threshold, self.token2token_id, self.token_id2token)
            kdr_tags = convert_typo_predictions_into_tags(kdr_predictions, kdr_probabilities, "R", *args)
            ins_tags = convert_typo_predictions_into_tags(ins_predictions, ins_probabilities, "I", *args)

            # the prediction of the first token (= [CLS]) is excluded.
            # the prediction of the dummy token at the end is used for insertion only.
            post_text = apply_edit_operations(example.pre_text, kdr_tags[1 : seq_len + 1], ins_tags[1 : seq_len + 2])
            post_texts.append(post_text)

        if batch_idx == len(dataloader) - 1:
            for texts in dataset.stash.values():
                post_texts.extend(texts)
            dataset.stash.clear()

        output_string = "".join(post_text + "\nEOD\n" for post_text in post_texts)
        self.write_output_string(output_string)