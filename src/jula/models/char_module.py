from statistics import mean
from typing import Any, Optional

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning.core.lightning import LightningModule
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizer

from jula.evaluators.word_segmenter import WordSegmenterMetric
from jula.models.models.char_encoder import CharEncoder
from jula.models.models.word_segmenter import WordSegmenter


class CharModule(LightningModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            hparams.model.model_name_or_path,
            **hydra.utils.instantiate(
                hparams.dataset.tokenizer_kwargs, _convert_="partial"
            ),
        )

        self.valid_corpora = list(hparams.dataset.valid.keys())
        self.test_corpora = list(hparams.dataset.test.keys())

        self.char_encoder: CharEncoder = CharEncoder(hparams, self.tokenizer)
        pretrained_model_config: PretrainedConfig = (
            self.char_encoder.pretrained_model.config
        )
        self.word_segmenter: WordSegmenter = WordSegmenter(
            hparams=hparams,
            pretrained_model_config=pretrained_model_config,
        )
        self.valid_word_segmenter_metrics: dict[str, WordSegmenterMetric] = {
            corpus: WordSegmenterMetric() for corpus in self.valid_corpora
        }
        self.test_word_segmenter_metrics: dict[str, WordSegmenterMetric] = {
            corpus: WordSegmenterMetric() for corpus in self.test_corpora
        }

    def forward(self, **kwargs) -> dict[str, dict[str, torch.Tensor]]:
        encoder_output = self.char_encoder(kwargs)  # (b, seq_len, h)
        word_segmenter_outputs = self.word_segmenter(encoder_output, kwargs)
        return {
            "word_segmenter_outputs": word_segmenter_outputs,
        }

    def training_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        outputs: dict[str, dict[str, torch.Tensor]] = self(**batch)
        self.log(
            "train/word_segmenter_loss",
            outputs["word_segmenter_outputs"]["loss"],
            on_step=True,
            on_epoch=True,
        )
        return outputs["word_segmenter_outputs"]["loss"]

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        outputs: dict[str, dict[str, torch.Tensor]] = self(**batch)
        corpus = self.valid_corpora[dataloader_idx or 0]
        word_segmenter_args = {
            "seg_preds": torch.argmax(
                outputs["word_segmenter_outputs"]["logits"], dim=-1
            ),
            "seg_labels": batch["seg_labels"],
        }
        self.valid_word_segmenter_metrics[corpus].update(**word_segmenter_args)
        self.log(
            "valid/word_segmenter_loss",
            outputs["word_segmenter_outputs"]["loss"],
        )

    def validation_epoch_end(self, validation_step_outputs) -> None:
        f1s: list[float] = []
        word_segmenter_f1 = 0.0
        for corpus, metric in self.valid_word_segmenter_metrics.items():
            for name, value in metric.compute().items():
                if name == "word_segmenter_f1":
                    word_segmenter_f1 += value
                self.log(f"valid_{corpus}/{name}", value)
                metric.reset()
        self.log(
            "valid/word_segmenter_f1",
            word_segmenter_f1 / len(self.valid_word_segmenter_metrics),
        )
        f1s.append(word_segmenter_f1 / len(self.valid_word_segmenter_metrics))

        self.log("valid/f1", mean(f1s))

    def test_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        outputs: dict[str, dict[str, torch.Tensor]] = self(**batch)
        corpus = self.test_corpora[dataloader_idx or 0]
        word_segmenter_args = {
            "seg_preds": torch.argmax(
                outputs["word_segmenter_outputs"]["logits"], dim=-1
            ),
            "seg_labels": batch["seg_labels"],
        }
        self.test_word_segmenter_metrics[corpus].update(**word_segmenter_args)
        self.log(
            "test/word_segmenter_loss",
            outputs["word_segmenter_outputs"]["loss"],
        )

    def test_epoch_end(self, test_step_outputs) -> None:
        f1s: list[float] = []
        word_segmenter_f1 = 0.0
        for corpus, metric in self.test_word_segmenter_metrics.items():
            for name, value in metric.compute().items():
                if name == "word_segmenter_f1":
                    word_segmenter_f1 += value
                self.log(f"test_{corpus}/{name}", value)
                metric.reset()
        self.log(
            "test/word_segmenter_f1",
            word_segmenter_f1 / len(self.test_word_segmenter_metrics),
        )
        f1s.append(word_segmenter_f1 / len(self.test_word_segmenter_metrics))

        self.log("test/f1", mean(f1s))

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        outputs: dict[str, dict[str, torch.Tensor]] = self(**batch)
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
