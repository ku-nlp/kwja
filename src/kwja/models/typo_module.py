from typing import Any, Optional, Union

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.core.lightning import LightningModule
from transformers import PretrainedConfig

from kwja.evaluators.typo_correction_metric import TypoCorrectionMetric
from kwja.models.models.char_encoder import CharEncoder
from kwja.models.models.typo_corrector import TypoCorrector
from kwja.utils.util import filter_dict_items


class TypoModule(LightningModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__()
        OmegaConf.resolve(hparams)
        self.save_hyperparameters(hparams)

        self.char_encoder: CharEncoder = CharEncoder(hparams)

        pretrained_model_config: PretrainedConfig = self.char_encoder.pretrained_model.config
        self.model: TypoCorrector = TypoCorrector(
            hparams=hparams,
            pretrained_model_config=pretrained_model_config,
        )
        self.metrics: TypoCorrectionMetric = TypoCorrectionMetric()

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
        # the prediction of the first token (= [CLS]) is excluded.
        kdr_probs = torch.softmax(outputs["kdr_logits"][:, 1:, :], dim=-1)  # (b, seq_len - 1, kdr_label_num)
        kdr_values, kdr_indices = torch.max(kdr_probs, dim=-1)
        ins_probs = torch.softmax(outputs["ins_logits"][:, 1:, :], dim=-1)  # (b, seq_len - 1, ins_label_num)
        ins_values, ins_indices = torch.max(ins_probs, dim=-1)
        return {
            "texts": batch["texts"],
            "kdr_values": kdr_values,
            "kdr_indices": kdr_indices,
            "ins_values": ins_values,
            "ins_indices": ins_indices,
        }

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

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        hparams = checkpoint["hyper_parameters"]["hparams"]
        OmegaConf.set_struct(hparams, False)
        hparams = filter_dict_items(hparams, self.hparams.hparams_to_ignore_on_save)
        checkpoint["hyper_parameters"] = {"hparams": hparams}
