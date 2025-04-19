from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Union

import lightning as L
from transformers import PreTrainedTokenizerBase

from kwja.callbacks.base_module_writer import BaseModuleWriter
from kwja.callbacks.utils import apply_edit_operations, convert_typo_predictions_into_tags
from kwja.datamodule.datasets import TypoDataset, TypoInferenceDataset
from kwja.datamodule.datasets.typo import get_maps
from kwja.datamodule.examples import TypoExample, TypoInferenceExample


class TypoModuleWriter(BaseModuleWriter):
    def __init__(
        self,
        confidence_threshold: float,
        tokenizer: PreTrainedTokenizerBase,
        destination: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__(destination=destination)
        self.confidence_threshold = confidence_threshold
        self.token2token_id, self.token_id2token = get_maps(tokenizer)

    def write_on_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,  # noqa: ARG002
        prediction: Any,
        batch_indices: Optional[Sequence[int]],  # noqa: ARG002
        batch: Any,  # noqa: ARG002
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if isinstance(trainer.predict_dataloaders, dict):
            dataloader = list(trainer.predict_dataloaders.values())[dataloader_idx]
        else:
            dataloader = trainer.predict_dataloaders[dataloader_idx]
        dataset: Union[TypoDataset, TypoInferenceDataset] = dataloader.dataset

        post_texts: list[str] = []
        doc_ids: list[str] = []
        for example_id, kdr_predictions, kdr_probabilities, ins_predictions, ins_probabilities in zip(
            *[v.tolist() for v in prediction.values()]
        ):
            if example_id in dataset.stash:
                texts = dataset.stash.pop(example_id)
                post_texts.extend([t[0] for t in texts])
                doc_ids.extend([t[1] for t in texts])

            example: Union[TypoExample, TypoInferenceExample] = dataset.examples[example_id]
            seq_len: int = len(example.pre_text)
            if seq_len == 0:
                continue

            args = (self.confidence_threshold, self.token_id2token)
            kdr_tags = convert_typo_predictions_into_tags(kdr_predictions, kdr_probabilities, "R", *args)
            ins_tags = convert_typo_predictions_into_tags(ins_predictions, ins_probabilities, "I", *args)

            # the prediction of the first token (= [CLS]) is excluded.
            # the prediction of the dummy token at the end is used for insertion only.
            post_text = apply_edit_operations(example.pre_text, kdr_tags[1 : seq_len + 1], ins_tags[1 : seq_len + 2])
            post_texts.append(post_text)
            doc_ids.append(example.doc_id)

        if batch_idx == len(dataloader) - 1:
            for texts in dataset.stash.values():
                post_texts.extend([t[0] for t in texts])
                doc_ids.extend([t[1] for t in texts])
            dataset.stash.clear()

        output_string = ""
        for text, doc_id in zip(post_texts, doc_ids):
            if doc_id != "":
                output_string += f"# D-ID:{doc_id}\n"
            output_string += text + "\nEOD\n"

        self.write_output_string(output_string)
