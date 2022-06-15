import json
import os
from typing import Any, Optional, Sequence

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from transformers import AutoTokenizer, PreTrainedTokenizer

from jula.evaluators.word_segment_metrics import WordSegmenterMetrics


class WordSegmenterWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        pred_filename: str = "predict",
        model_name_or_path: str = "cl-tohoku/bert-base-japanese-char",
        tokenizer_kwargs: dict = None,
    ) -> None:
        super().__init__(write_interval="epoch")
        self.output_path = f"{output_dir}/{pred_filename}.json"
        if os.path.isfile(self.output_path):
            os.remove(self.output_path)

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **tokenizer_kwargs,
        )
        self.predicts: dict[int, Any] = dict()
        self.metrics = WordSegmenterMetrics()

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

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
    ) -> None:
        example_id = 0
        for prediction in predictions:
            for batch_pred in prediction:
                seg_preds, seg_labels = self.metrics.convert_num2label(
                    preds=torch.argmax(batch_pred["logits"], dim=-1).cpu().tolist(),
                    labels=batch_pred["seg_labels"].cpu().tolist(),
                )  # (b, seq_len), (b, seq_len)
                for idx in range(len(batch_pred["input_ids"])):
                    self.predicts[example_id] = dict(
                        input_ids=self.tokenizer.decode(
                            [
                                x
                                for x in batch_pred["input_ids"][idx]
                                if x != self.tokenizer.pad_token_id
                            ]
                        ),
                        seg_preds=seg_preds[idx],
                        seg_labels=seg_labels[idx],
                    )
                    example_id += 1
        with open(self.output_path, "w") as f:
            json.dump(self.predicts, f, ensure_ascii=False, indent=2)
