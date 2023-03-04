from statistics import mean
from typing import Any, Dict, List, Optional

import hydra
import torch
from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.generation import LogitsProcessorList

from kwja.metrics import Seq2SeqModuleMetric
from kwja.modules.base import BaseModule
from kwja.modules.components.logits_processor import ForcedSurfLogitsProcessor, get_char2tokens
from kwja.utils.constants import IGNORE_INDEX


class Seq2SeqModule(BaseModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__(hparams)

        if valid_corpora := getattr(hparams.datamodule, "valid", None):
            self.valid_corpora: List[str] = list(valid_corpora)
            self.valid_corpus2seq2seq_module_metric: Dict[str, Seq2SeqModuleMetric] = {
                corpus: Seq2SeqModuleMetric() for corpus in self.valid_corpora
            }
        if test_corpora := getattr(hparams.datamodule, "test", None):
            self.test_corpora: List[str] = list(test_corpora)
            self.test_corpus2seq2seq_module_metric: Dict[str, Seq2SeqModuleMetric] = {
                corpus: Seq2SeqModuleMetric() for corpus in self.test_corpora
            }

        self.tokenizer: PreTrainedTokenizerBase = hydra.utils.call(hparams.module.tokenizer)

        self.seq2seq_model: PreTrainedModel = hydra.utils.call(hparams.encoder)
        if hasattr(hparams, "special_tokens"):
            # https://github.com/huggingface/transformers/issues/4875
            self.seq2seq_model.resize_token_embeddings(len(self.tokenizer.get_vocab()))

        self.char2tokens, self.char2underscore_tokens = get_char2tokens(self.tokenizer)
        self.use_forced_surf_decoding: bool = getattr(hparams, "use_forced_surf_decoding", False)

    def forward(self, batch: Any) -> Dict[str, torch.Tensor]:
        self._truncate(batch)
        output = self.seq2seq_model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["seq2seq_labels"]
        )
        return {"loss": output.loss, "logits": output.logits}

    @staticmethod
    def _truncate(batch: Any) -> None:
        max_src_length = batch["attention_mask"].sum(dim=1).max().item()
        for key, value in batch.items():
            if key in {"input_ids", "attention_mask"}:
                batch[key] = value[:, :max_src_length].contiguous()
        max_tgt_length = (batch["seq2seq_labels"] != IGNORE_INDEX).sum(dim=1).max().item()
        batch["seq2seq_labels"] = batch["seq2seq_labels"][:, :max_tgt_length].contiguous()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        ret: Dict[str, torch.Tensor] = self(batch)
        self.log("train/loss", ret["loss"])
        return ret["loss"]

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> None:
        kwargs = self(batch)
        kwargs.update({"example_ids": batch["example_ids"], "loss": kwargs["loss"]})
        corpus = self.valid_corpora[dataloader_idx or 0]
        self.valid_corpus2seq2seq_module_metric[corpus].update(kwargs)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        metrics_log: Dict[str, Dict[str, float]] = {corpus: {} for corpus in self.valid_corpora}
        for corpus, seq2seq_module_metric in self.valid_corpus2seq2seq_module_metric.items():
            metrics_log[corpus] = seq2seq_module_metric.compute()
            seq2seq_module_metric.reset()

        for corpus, metrics in metrics_log.items():
            self.log_dict({f"valid_{corpus}/{key}": value for key, value in metrics.items()})
        for key in list(metrics_log.values())[0].keys():
            mean_score = mean(metrics_log[corpus][key] for corpus in self.valid_corpora if key in metrics_log[corpus])
            self.log(f"valid/{key}", mean_score)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> None:
        kwargs = self(batch)
        kwargs.update({"example_ids": batch["example_ids"], "loss": kwargs["loss"]})
        corpus = self.test_corpora[dataloader_idx or 0]
        self.test_corpus2seq2seq_module_metric[corpus].update(kwargs)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        metrics_log: Dict[str, Dict[str, float]] = {corpus: {} for corpus in self.test_corpora}
        for corpus, seq2seq_module_metric in self.test_corpus2seq2seq_module_metric.items():
            metrics_log[corpus] = seq2seq_module_metric.compute()
            seq2seq_module_metric.reset()

        for corpus, metrics in metrics_log.items():
            self.log_dict({f"test_{corpus}/{key}": value for key, value in metrics.items()})
        for key in list(metrics_log.values())[0].keys():
            mean_score = mean(metrics_log[corpus][key] for corpus in self.test_corpora if key in metrics_log[corpus])
            self.log(f"test/{key}", mean_score)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        generateds = self.seq2seq_model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            logits_processor=LogitsProcessorList(
                [
                    ForcedSurfLogitsProcessor(
                        texts=batch["src_text"],
                        tokenizer=self.tokenizer,
                        char2tokens=self.char2tokens,
                        char2underscore_tokens=self.char2underscore_tokens,
                    ),
                ]
            )
            if self.use_forced_surf_decoding
            else None,
            **self.hparams.decoding,
        )  # (b, max_tgt_len)
        return {
            "example_ids": batch["example_ids"] if "example_ids" in batch else [],
            "seq2seq_predictions": generateds["sequences"],
        }
