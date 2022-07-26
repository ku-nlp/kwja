import os
from typing import Any, Optional, Sequence

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from jula.utils.constants import INDEX2SEG_TYPE


class WordModuleWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        pred_filename: str = "predict",
        model_name_or_path: str = "cl-tohoku/bert-base-japanese-char",
        tokenizer_kwargs: dict = None,
        use_stdout: bool = False,
    ) -> None:
        super().__init__(write_interval="epoch")

        self.use_stdout = use_stdout
        if self.use_stdout:
            self.output_path = ""
        else:
            self.output_path = f"{output_dir}/{pred_filename}.txt"
            os.makedirs(output_dir, exist_ok=True)
            if os.path.isfile(self.output_path):
                os.remove(self.output_path)

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

        out = "\n".join(results)
        if self.use_stdout:
            print(out)
        else:
            with open(self.output_path, "w") as f:
                f.write(out)
