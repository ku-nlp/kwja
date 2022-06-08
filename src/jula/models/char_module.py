from typing import Any, Optional, Union

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning.core.lightning import LightningModule
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from jula.evaluators.typo_corrector_metrics import TypoCorrectorMetrics
from jula.evaluators.word_segment_metrics import WordSegmenterMetrics
from jula.models.models.typo_corrector import TypoCorrector
from jula.models.models.word_segmenter import WordSegmenter


class CharModule(LightningModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            hparams.model.model_name_or_path,
            **hparams.dataset.tokenizer_kwargs,
        )

        if hparams.module.type == "char":
            self.model: WordSegmenter = WordSegmenter(
                hparams=hparams, tokenizer=self.tokenizer
            )
            self.metrics: WordSegmenterMetrics = WordSegmenterMetrics()
        elif hparams.module.type == "char_typo":
            self.model: TypoCorrector = TypoCorrector(
                hparams=hparams, tokenizer=self.tokenizer
            )
            self.metrics: TypoCorrector = TypoCorrectorMetrics()
        else:
            raise ValueError("invalid module type")

    def forward(self, **kwargs):
        return self.model(kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        outputs: dict[str, torch.Tensor] = self(**batch)
        result: dict[str, Union[torch.Tensor, float]] = self.metrics.compute_metrics(
            outputs=outputs,
            batch=batch,
        )
        for name, value in result.items():
            self.log(f"train/{name}", value)
        return result

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> dict[str, Any]:
        outputs: dict[str, torch.Tensor] = self(**batch)
        result: dict[str, Union[torch.Tensor, float]] = self.metrics.compute_metrics(
            outputs=outputs,
            batch=batch,
        )
        for name, value in result.items():
            self.log(f"valid/{name}", value)
        return result

    def test_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> dict[str, Any]:
        outputs: dict[str, torch.Tensor] = self(**batch)
        result: dict[str, Union[torch.Tensor, float]] = self.metrics.compute_metrics(
            outputs=outputs,
            batch=batch,
        )
        for name, value in result.items():
            self.log(f"test/{name}", value)
        return result

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        outputs: dict[str, torch.Tensor] = self(**batch)
        predict_output = dict(input_ids=batch["input_ids"])
        for key in batch:
            if "labels" in key:
                predict_output[key] = batch[key]
        for key in outputs:
            if "logits" in key:
                predict_output[key] = outputs[key]
        return predict_output

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, params=self.parameters(), _convert_="partial"
        )
        return [optimizer], []
