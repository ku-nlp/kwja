import os
import sys
from io import TextIOBase
from pathlib import Path
from typing import Any, Optional, Sequence, TextIO, Union

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from jula.utils.constants import INDEX2SEG_TYPE


class CharModuleWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        pred_filename: str = "predict",
        model_name_or_path: str = "cl-tohoku/bert-base-japanese-char",
        tokenizer_kwargs: dict = None,
        use_stdout: bool = False,
    ) -> None:
        super().__init__(write_interval="epoch")

        self.destination = Union[Path, TextIO]
        if use_stdout is True:
            self.destination = sys.stdout
        else:
            self.destination = Path(f"{output_dir}/{pred_filename}.txt")
            self.destination.parent.mkdir(exist_ok=True)
            if self.destination.exists():
                os.remove(str(self.destination))

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **hydra.utils.instantiate(tokenizer_kwargs or {}, _convert_="partial"),
        )

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]] = None,
    ) -> None:
        results = []
        for prediction in predictions:
            for batch_pred in prediction:
                batch_size = len(batch_pred["input_ids"])
                for i in range(batch_size):
                    input_ids = batch_pred["input_ids"][i].cpu().tolist()  # (seq_len,)
                    pred_logits = batch_pred["word_segmenter_logits"][i]  # (seq_len, len(INDEX2SEG_TYPE))
                    pred_ids = torch.argmax(pred_logits, dim=1).cpu().tolist()  # (seq_len,)
                    pred_types = [INDEX2SEG_TYPE[id_] for id_ in pred_ids]  # (seq_len,)
                    assert len(input_ids) == len(pred_types)
                    result = ""
                    for input_id, pred_type in zip(input_ids, pred_types):
                        if input_id in self.tokenizer.all_special_ids:
                            continue
                        if pred_type == "B":
                            result += " "
                        result += self.tokenizer.decode(input_id)
                    results.append(result.strip())

        output_string: str = "\n".join(results) + "\n"
        if isinstance(self.destination, Path):
            self.destination.write_text(output_string)
        elif isinstance(self.destination, TextIOBase):
            self.destination.write(output_string)
