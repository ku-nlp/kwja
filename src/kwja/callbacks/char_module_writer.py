import os
import sys
from io import TextIOBase
from pathlib import Path
from typing import Any, Optional, Sequence, TextIO, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter

from kwja.datamodule.datasets import CharDataset, CharInferenceDataset
from kwja.datamodule.datasets.char_dataset import CharExampleSet
from kwja.datamodule.datasets.char_inference_dataset import CharInferenceExample
from kwja.utils.constants import INDEX2SEG_TYPE, INDEX2WORD_NORM_TYPE
from kwja.utils.word_normalize import get_normalized


class CharModuleWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        pred_filename: str = "predict",
        use_stdout: bool = False,
    ) -> None:
        super().__init__(write_interval="epoch")

        self.destination = Union[Path, TextIO]
        if use_stdout is True:
            self.destination = sys.stdout
        else:
            self.destination = Path(f"{output_dir}/{pred_filename}.txt")
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
        # TODO: Write out results with the original surface form
        results = []
        dataloaders = trainer.predict_dataloaders
        for prediction in predictions:
            for batch_pred in prediction:
                dataset: Union[CharDataset, CharInferenceDataset] = dataloaders[batch_pred["dataloader_idx"]].dataset
                special_ids = {
                    special_id
                    for special_id in dataset.tokenizer.all_special_ids
                    if special_id != dataset.tokenizer.unk_token_id
                }
                batch_size = len(batch_pred["input_ids"])
                for i in range(batch_size):
                    example: Union[CharExampleSet, CharInferenceExample] = dataset.examples[
                        batch_pred["example_ids"][i]
                    ]
                    input_ids = batch_pred["input_ids"][i].cpu().tolist()  # (seq_len,)
                    word_segmenter_logits = batch_pred["word_segmenter_logits"][i]  # (seq_len, len(INDEX2SEG_TYPE))
                    word_segmenter_preds = torch.argmax(word_segmenter_logits, dim=1).cpu().tolist()  # (seq_len,)
                    word_normalizer_logits = batch_pred["word_normalizer_logits"][i]  # (seq_len, len(INDEX2SEG_TYPE))
                    word_normalizer_preds = torch.argmax(word_normalizer_logits, dim=1).cpu().tolist()  # (seq_len,)
                    assert len(input_ids) == len(word_segmenter_preds) == len(word_normalizer_preds)

                    word_segmenter_types = [
                        INDEX2SEG_TYPE[pred_id]
                        for input_id, pred_id in zip(input_ids, word_segmenter_preds)
                        if input_id not in special_ids
                    ]
                    word_normalizer_types = [
                        INDEX2WORD_NORM_TYPE[pred_id]
                        for input_id, pred_id in zip(input_ids, word_normalizer_preds)
                        if input_id not in special_ids
                    ]
                    char_idx = 0
                    document = dataset.doc_id2document[example.doc_id]
                    for sentence in document.sentences:
                        results.append(sentence.comment)
                        result: str = ""
                        word_surf: str = ""
                        word_norm_ops: list[str] = []
                        for char in sentence.text:
                            if word_segmenter_types[char_idx] == "B" and word_surf:
                                if result:
                                    result += " "
                                result += get_normalized(word_surf, word_norm_ops, strict=False)
                                word_surf = ""
                                word_norm_ops = []
                            word_surf += char
                            word_norm_ops.append(word_normalizer_types[char_idx])
                            char_idx += 1
                        if result:
                            result += " "
                        result += get_normalized(word_surf, word_norm_ops, strict=False)
                        results.append(result)

        output_string: str = "\n".join(results) + "\n"
        if isinstance(self.destination, Path):
            self.destination.write_text(output_string)
        elif isinstance(self.destination, TextIOBase):
            self.destination.write(output_string)

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
        pass
