import json
import os
from collections import defaultdict
from typing import Any, Optional, Sequence

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from jula.utils.utils import BASE_PHRASE_FEATURES


class PhraseAnalysisWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        pred_filename: str = "predict",
        model_name_or_path: str = "nlp-waseda/roberta-base-japanese",
        tokenizer_kwargs: dict = None,
    ) -> None:
        super().__init__(write_interval="epoch")
        self.output_path = f"{output_dir}/{pred_filename}.json"
        if os.path.isfile(self.output_path):
            os.remove(self.output_path)

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **hydra.utils.instantiate(tokenizer_kwargs, _convert_="partial"),
        )
        self.pad_token_id = self.tokenizer.pad_token_id

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
                batch_preds = torch.where(
                    prediction_step_output["phrase_analysis_logits"] >= 0.5, 1.0, 0.0
                )
                for document_id, preds, labels in zip(
                    prediction_step_output["document_ids"],
                    batch_preds.tolist(),
                    prediction_step_output["base_phrase_features"].tolist(),
                ):
                    document = dataset.documents[document_id]
                    results[corpus].append(
                        [
                            f"{morpheme.surf}|{''.join(self.convert(pred))}|{''.join(self.convert(label))}"
                            for morpheme, pred, label in zip(
                                document.morphemes, preds, labels
                            )
                        ]
                    )

        with open(self.output_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    @staticmethod
    def convert(vector):
        return [
            f"<{feature}>"
            for feature, element in zip(BASE_PHRASE_FEATURES, vector)
            if element == 1.0
        ]
