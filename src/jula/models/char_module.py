from typing import Any, Optional

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase

from jula.evaluators.word_normalization_metric import WordNormalizationMetric
from jula.evaluators.word_segmentation_metric import WordSegmentationMetric
from jula.models.models.char_encoder import CharEncoder
from jula.models.models.word_normalizer import WordNormalizer
from jula.models.models.word_segmenter import WordSegmenter


class CharModule(LightningModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()

        self.valid_corpora = list(hparams.datamodule.valid.keys())
        self.test_corpora = list(hparams.datamodule.test.keys())

        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            hparams.model.model_name_or_path,
            **hydra.utils.instantiate(hparams.dataset.tokenizer_kwargs),
        )
        self.char_encoder = CharEncoder(hparams, vocab_size=len(tokenizer.get_vocab()))

        pretrained_model_config: PretrainedConfig = self.char_encoder.pretrained_model.config

        self.word_segmenter = WordSegmenter(hparams, pretrained_model_config)
        self.valid_word_segmenter_metrics: dict[str, WordSegmentationMetric] = {
            corpus: WordSegmentationMetric() for corpus in self.valid_corpora
        }
        self.test_word_segmenter_metrics: dict[str, WordSegmentationMetric] = {
            corpus: WordSegmentationMetric() for corpus in self.test_corpora
        }

        self.word_normalizer = WordNormalizer(hparams, pretrained_model_config)
        self.valid_word_normalizer_metrics: dict[str, WordNormalizationMetric] = {
            corpus: WordNormalizationMetric() for corpus in self.valid_corpora
        }
        self.test_word_normalizer_metrics: dict[str, WordNormalizationMetric] = {
            corpus: WordNormalizationMetric() for corpus in self.test_corpora
        }

    def forward(self, **kwargs) -> dict[str, dict[str, torch.Tensor]]:
        encoder_output = self.char_encoder(kwargs)  # (b, seq_len, h)
        word_segmenter_outputs = self.word_segmenter(encoder_output, kwargs)
        word_normalizer_outputs = self.word_normalizer(encoder_output, kwargs)
        return {
            "word_segmenter_outputs": word_segmenter_outputs,
            "word_normalizer_outputs": word_normalizer_outputs,
        }

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        outputs: dict[str, dict[str, torch.Tensor]] = self(**batch)
        word_segmenter_loss = outputs["word_segmenter_outputs"]["loss"]
        self.log("train/word_segmenter_loss", word_segmenter_loss)
        word_normalizer_loss = outputs["word_normalizer_outputs"]["loss"]
        self.log("train/word_normalizer_loss", word_normalizer_loss)
        return word_segmenter_loss + word_normalizer_loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        outputs: dict[str, dict[str, torch.Tensor]] = self(**batch)
        corpus = self.valid_corpora[dataloader_idx or 0]
        word_segmenter_args = {
            "seg_preds": torch.argmax(outputs["word_segmenter_outputs"]["logits"], dim=-1),
            "seg_labels": batch["seg_labels"],
        }
        self.valid_word_segmenter_metrics[corpus].update(**word_segmenter_args)
        self.log("valid/word_segmenter_loss", outputs["word_segmenter_outputs"]["loss"])
        word_normalizer_args = {
            "charnorm_preds": torch.argmax(outputs["word_normalizer_outputs"]["logits"], dim=-1),
            "charnorm_labels": batch["charnorm_labels"],
        }
        self.valid_word_normalizer_metrics[corpus].update(**word_normalizer_args)
        self.log("valid/word_normalizer_loss", outputs["word_normalizer_outputs"]["loss"])

    def validation_epoch_end(self, validation_step_outputs) -> None:
        word_segmenter_f1 = 0.0
        for corpus, metric in self.valid_word_segmenter_metrics.items():
            for name, value in metric.compute().items():
                if name == "word_segmenter_f1":
                    word_segmenter_f1 += value
                self.log(f"valid_{corpus}/{name}", value)
                metric.reset()
        word_segmenter_f1 /= len(self.valid_word_segmenter_metrics)
        self.log("valid/word_segmenter_f1", word_segmenter_f1)
        self.log("valid/f1", word_segmenter_f1)

        word_normalizer_f1 = 0.0
        for corpus, metric in self.valid_word_normalizer_metrics.items():
            for name, value in metric.compute().items():
                if name == "word_segmenter_f1":
                    word_segmenter_f1 += value
                self.log(f"valid_{corpus}/{name}", value)
                metric.reset()
        word_normalizer_f1 /= len(self.valid_word_normalizer_metrics)
        self.log("valid/word_normalizer_f1", word_normalizer_f1)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        outputs: dict[str, dict[str, torch.Tensor]] = self(**batch)
        corpus = self.test_corpora[dataloader_idx or 0]
        word_segmenter_args = {
            "seg_preds": torch.argmax(outputs["word_segmenter_outputs"]["logits"], dim=-1),
            "seg_labels": batch["seg_labels"],
        }
        self.test_word_segmenter_metrics[corpus].update(**word_segmenter_args)
        self.log("test/word_segmenter_loss", outputs["word_segmenter_outputs"]["loss"])
        word_normalizer_args = {
            "charnorm_preds": torch.argmax(outputs["word_normalizer_outputs"]["logits"], dim=-1),
            "charnorm_labels": batch["charnorm_labels"],
        }
        self.test_word_normalizer_metrics[corpus].update(**word_normalizer_args)
        self.log("test/word_normalizer_loss", outputs["word_normalizer_outputs"]["loss"])

    def test_epoch_end(self, test_step_outputs) -> None:
        word_segmenter_f1 = 0.0
        for corpus, metric in self.test_word_segmenter_metrics.items():
            for name, value in metric.compute().items():
                if name == "word_segmenter_f1":
                    word_segmenter_f1 += value
                self.log(f"test_{corpus}/{name}", value)
                metric.reset()
        word_segmenter_f1 /= len(self.test_word_segmenter_metrics)
        self.log("test/word_segmenter_f1", word_segmenter_f1)
        self.log("test/f1", word_segmenter_f1)

        word_normalizer_f1 = 0.0
        for corpus, metric in self.test_word_normalizer_metrics.items():
            for name, value in metric.compute().items():
                if name == "word_segmenter_f1":
                    word_segmenter_f1 += value
                self.log(f"test_{corpus}/{name}", value)
                metric.reset()
        word_normalizer_f1 /= len(self.test_word_normalizer_metrics)
        self.log("valid/word_normalizer_f1", word_normalizer_f1)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        outputs: dict[str, dict[str, torch.Tensor]] = self(**batch)
        return {
            "dataloader_idx": dataloader_idx or 0,
            "input_ids": batch["input_ids"],
            "word_segmenter_logits": outputs["word_segmenter_outputs"]["logits"],
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
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step", "frequency": 1},
        }
