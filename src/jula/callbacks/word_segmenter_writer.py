import os
from typing import Any, Optional, Sequence

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from transformers import AutoTokenizer, PreTrainedTokenizer

from jula.evaluators.word_segmenter import WordSegmenterMetric


class WordSegmenterWriter(BasePredictionWriter):
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
            self.output_path = f"{output_dir}/{pred_filename}.json"
            os.makedirs(output_dir, exist_ok=True)
            if os.path.isfile(self.output_path):
                os.remove(self.output_path)

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **tokenizer_kwargs,
        )
        self.metrics: WordSegmenterMetric = WordSegmenterMetric()

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
    ) -> None:
        results = []
        for prediction in predictions:
            for batch_pred in prediction:
                seg_preds = [
                    self.metrics.convert_ids_to_labels(ids)
                    for ids in torch.argmax(batch_pred["logits"], dim=-1).cpu().tolist()
                ]  # (b, seq_len)
                for item_index in range(len(batch_pred["input_ids"])):
                    result = ""
                    for token_index in range(len(batch_pred["input_ids"][item_index])):
                        token_id = batch_pred["input_ids"][item_index][token_index]
                        if token_id in self.tokenizer.all_special_ids:
                            continue
                        seg_pred = seg_preds[item_index][token_index]
                        if seg_pred == "B":
                            result += " "
                        result += self.tokenizer.decode(token_id)
                    results.append(result.strip())
        if self.use_stdout:
            print("\n".join(results))
        else:
            with open(self.output_path, "w") as f:
                f.write("\n".join(results))
