from typing import Any, Optional, Union

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.core.lightning import LightningModule
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from jula.evaluators.phrase_analysis_metrics import PhraseAnalysisMetrics
from jula.models.models.word_encoder import WordEncoder
from jula.models.models.phrase_analyzer import PhraseAnalyzer
from jula.models.models.pooling import PoolingStrategy


class WordModule(LightningModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            hparams.model.model_name_or_path,
            **OmegaConf.to_container(hparams.dataset.tokenizer_kwargs),
        )

        self.word_encoder: WordEncoder = WordEncoder(hparams, self.tokenizer)

        self.phrase_analyzer: PhraseAnalyzer = PhraseAnalyzer(
            hparams=hparams, pretrained_model_config=self.word_encoder.pretrained_model.config
        )
        self.phrase_analysis_metrics: PhraseAnalysisMetrics = PhraseAnalysisMetrics()

    def forward(self, **batch):
        encoder_output = self.word_encoder(batch)  # (b, seq_len, h)
        phrase_analyzer_output = self.phrase_analyzer(
            encoder_output, batch, PoolingStrategy.FIRST
        )
        return {'phrase_analyzer_output': phrase_analyzer_output}

    def training_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        output: dict[str, torch.Tensor] = self(**batch)
        phrase_analysis_result: dict[str, Union[torch.Tensor, float]] = self.phrase_analysis_metrics.compute_metrics(
            output=output['phrase_analyzer_output'],
            batch=batch
        )
        for name, value in phrase_analysis_result.items():
            self.log(f"train/{name}", value, on_step=True, on_epoch=True)
        return output['phrase_analyzer_output']['phrase_analysis_loss']

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> dict[str, Any]:
        output: dict[str, torch.Tensor] = self(**batch)
        phrase_analysis_result: dict[str, Union[torch.Tensor, float]] = self.phrase_analysis_metrics.compute_metrics(
            output=output['phrase_analyzer_output'],
            batch=batch,
        )
        for name, value in phrase_analysis_result.items():
            self.log(f"valid/{name}", value, on_step=True, on_epoch=True)
        return phrase_analysis_result

    def test_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> dict[str, Any]:
        output: dict[str, torch.Tensor] = self(**batch)
        phrase_analysis_result: dict[str, Union[torch.Tensor, float]] = self.phrase_analysis_metrics.compute_metrics(
            output=output['phrase_analyzer_output'],
            batch=batch,
        )
        for name, value in phrase_analysis_result.items():
            self.log(f"test/{name}", value, on_step=True, on_epoch=True)
        return phrase_analysis_result

    def predict_step(
            self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, params=self.parameters(), _convert_="partial"
        )
        return [optimizer], []
