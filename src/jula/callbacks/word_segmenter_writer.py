import json
import os
from collections import defaultdict
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
    ) -> None:
        super().__init__(write_interval="epoch")
        self.output_path = f"{output_dir}/{pred_filename}.json"

        os.makedirs(output_dir, exist_ok=True)
        if os.path.isfile(self.output_path):
            os.remove(self.output_path)

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **tokenizer_kwargs,
        )
        self.metrics: WordSegmenterMetric = WordSegmenterMetric()

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
        results = defaultdict(list)
        for dataloader_idx, prediction_step_outputs in enumerate(predictions):
            corpus = pl_module.test_corpora[dataloader_idx]
            dataset = trainer.datamodule.test_datasets[corpus]
            for prediction_step_output in prediction_step_outputs:
                batch_seg_preds, batch_seg_labels = self.metrics.convert_num2label(
                    preds=torch.argmax(
                        prediction_step_output["word_segmenter_logits"], dim=-1
                    ).tolist(),
                    labels=prediction_step_output["word_segmenter_labels"].tolist(),
                )  # (b, seq_len), (b, seq_len)
                for (document_id, seg_preds, seg_labels,) in zip(
                    prediction_step_output["document_ids"],
                    batch_seg_preds,
                    batch_seg_labels,
                ):
                    document = dataset.documents[document_id]
                    results[corpus].append(
                        {
                            "text": document.text,
                            "seg_preds": "".join(seg_preds),
                            "seg_labels": "".join(seg_labels),
                        }
                    )
        with open(self.output_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
