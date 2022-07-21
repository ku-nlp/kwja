from typing import Any, Optional, Union

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning.core.lightning import LightningModule
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizer

from jula.evaluators.typo_corrector import TypoCorrectorMetric
from jula.models.models.char_encoder import CharEncoder
from jula.models.models.typo_corrector import TypoCorrector


class TypoModule(LightningModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            hparams.model.model_name_or_path,
            **hydra.utils.instantiate(hparams.dataset.tokenizer_kwargs, _convert_="partial"),
        )

        self.char_encoder: CharEncoder = CharEncoder(hparams, self.tokenizer)
        pretrained_model_config: PretrainedConfig = self.char_encoder.pretrained_model.config
        self.model: TypoCorrector = TypoCorrector(
            hparams=hparams,
            pretrained_model_config=pretrained_model_config,
            tokenizer=self.tokenizer,
        )
        self.metrics: TypoCorrectorMetric = TypoCorrectorMetric()

    def forward(self, **kwargs):
        encoder_output = self.char_encoder(kwargs)  # (b, seq_len, h)
        model_output = self.model(encoder_output, kwargs)
        return model_output

    def training_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        outputs: dict[str, torch.Tensor] = self(**batch)
        result: dict[str, Union[torch.Tensor, float]] = self.metrics.compute_metrics(outputs, batch)
        for name, value in result.items():
            self.log(f"train/{name}", value)
        return result

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> dict[str, Any]:
        outputs: dict[str, torch.Tensor] = self(**batch)
        result: dict[str, Union[torch.Tensor, float]] = self.metrics.compute_metrics(outputs, batch)
        for name, value in result.items():
            self.log(f"valid/{name}", value)
        return result

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> dict[str, Any]:
        outputs: dict[str, torch.Tensor] = self(**batch)
        result: dict[str, Union[torch.Tensor, float]] = self.metrics.compute_metrics(outputs, batch)
        for name, value in result.items():
            self.log(f"test/{name}", value)
        return result

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
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
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ("bias", "LayerNorm.weight")
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.hparams.optimizer.weight_decay,
                "name": "decay",
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
                "name": "no_decay",
            },
        ]
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer,
            params=optimizer_grouped_parameters,
            _convert_="partial",
        )

        warmup_steps = self.hparams.warmup_steps
        lr_scheduler = hydra.utils.instantiate(
            self.hparams.scheduler,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
