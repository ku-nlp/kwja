import copy
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase

from kwja.evaluators.typo_module_metric import TypoModuleMetric
from kwja.models.components.head import SequenceLabelingHead
from kwja.utils.loss import compute_sequence_mean_loss
from kwja.utils.omegaconf import filter_dict_items


class TypoModule(pl.LightningModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)

        if valid_corpora := getattr(hparams.datamodule, "valid", None):
            self.valid_corpora: List[str] = list(valid_corpora)
            self.valid_corpus2typo_module_metric: Dict[str, TypoModuleMetric] = {
                corpus: TypoModuleMetric() for corpus in self.valid_corpora
            }
        if test_corpora := getattr(hparams.datamodule, "test", None):
            self.test_corpora: List[str] = list(test_corpora)
            self.test_corpus2typo_module_metric: Dict[str, TypoModuleMetric] = {
                corpus: TypoModuleMetric() for corpus in self.test_corpora
            }

        self.char_encoder: PreTrainedModel = hydra.utils.call(hparams.encoder)
        pretrained_model_config: PretrainedConfig = self.char_encoder.config
        if hasattr(hparams.dataset, "special_tokens"):
            self.char_encoder.resize_token_embeddings(
                pretrained_model_config.vocab_size + len(hparams.dataset.special_tokens)
            )
        head_args: Tuple[int, float] = (
            pretrained_model_config.hidden_size,
            pretrained_model_config.hidden_dropout_prob,
        )

        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            hparams.encoder.pretrained_model_name_or_path,
            **hydra.utils.instantiate(hparams.dataset.tokenizer_kwargs, _convert_="partial"),
        )
        self.kdr_tagger: SequenceLabelingHead = SequenceLabelingHead(len(tokenizer), *head_args)
        self.ins_tagger: SequenceLabelingHead = SequenceLabelingHead(
            len(tokenizer) + hparams.dataset.extended_vocab_size, *head_args
        )

    def forward(self, batch: Any) -> Dict[str, torch.Tensor]:
        truncation_length = self._truncate(batch)
        encoded = self.char_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        kdr_logits = self.kdr_tagger(encoded.last_hidden_state)
        ins_logits = self.ins_tagger(encoded.last_hidden_state)
        if truncation_length > 0:
            kdr_logits = self._pad(kdr_logits, truncation_length)
            ins_logits = self._pad(ins_logits, truncation_length)
        return {
            "kdr_logits": kdr_logits,
            "ins_logits": ins_logits,
        }

    def _truncate(self, batch: Any) -> int:
        max_seq_length = batch["attention_mask"].sum(dim=1).max().item()
        for key, value in batch.items():
            if key in {"input_ids", "attention_mask"}:
                batch[key] = value[:, :max_seq_length].contiguous()
        return self.hparams.max_seq_length - max_seq_length

    @staticmethod
    def _pad(tensor: torch.Tensor, padding_length: int) -> torch.Tensor:
        batch_size, _, num_labels = tensor.shape
        padding = torch.zeros((batch_size, padding_length, num_labels), dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding], dim=1)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        ret: Dict[str, torch.Tensor] = self(batch)
        kdr_loss = compute_sequence_mean_loss(ret["kdr_logits"], batch["kdr_labels"])
        self.log("train/kdr_loss", kdr_loss)
        ins_loss = compute_sequence_mean_loss(ret["ins_logits"], batch["ins_labels"])
        self.log("train/ins_loss", ins_loss)
        return kdr_loss + ins_loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> None:
        kwargs = self.predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        kwargs.update({"kdr_labels": batch["kdr_labels"], "ins_labels": batch["ins_labels"]})
        corpus = self.valid_corpora[dataloader_idx or 0]
        self.valid_corpus2typo_module_metric[corpus].update(kwargs)

    def validation_epoch_end(self, validation_step_outputs) -> None:
        metrics_log: Dict[str, Dict[str, float]] = {corpus: {} for corpus in self.valid_corpora}
        for corpus, typo_module_metric in self.valid_corpus2typo_module_metric.items():
            dataset = self.trainer.val_dataloaders[self.valid_corpora.index(corpus)].dataset
            typo_module_metric.set_properties(dataset)
            metrics = typo_module_metric.compute()
            metrics_log[corpus] = metrics
            typo_module_metric.reset()

        for corpus, metrics in metrics_log.items():
            self.log_dict({f"valid_{corpus}/{name}": value for name, value in metrics.items()})
        for name in list(metrics_log.values())[0].keys():
            self.log(f"valid/{name}", mean(metrics_log[corpus].get(name, 0) for corpus in self.valid_corpora))

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> None:
        kwargs = self.predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        kwargs.update({"kdr_labels": batch["kdr_labels"], "ins_labels": batch["ins_labels"]})
        corpus = self.test_corpora[dataloader_idx or 0]
        self.test_corpus2typo_module_metric[corpus].update(kwargs)

    def test_epoch_end(self, test_step_outputs) -> None:
        metrics_log: Dict[str, Dict[str, float]] = {corpus: {} for corpus in self.test_corpora}
        for corpus, typo_module_metric in self.test_corpus2typo_module_metric.items():
            dataset = self.trainer.test_dataloaders[self.test_corpora.index(corpus)].dataset
            typo_module_metric.set_properties(dataset)
            metrics = typo_module_metric.compute()
            metrics_log[corpus] = metrics
            typo_module_metric.reset()

        for corpus, metrics in metrics_log.items():
            self.log_dict({f"test_{corpus}/{name}": value for name, value in metrics.items()})
        for name in list(metrics_log.values())[0].keys():
            self.log(f"test/{name}", mean(metrics_log[corpus].get(name, 0) for corpus in self.test_corpora))

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        ret: Dict[str, torch.Tensor] = self(batch)
        kdr_probabilities = ret["kdr_logits"].softmax(dim=2)
        kdr_max_probabilities, kdr_predictions = kdr_probabilities.max(dim=2)
        ins_probabilities = ret["ins_logits"].softmax(dim=2)
        ins_max_probabilities, ins_predictions = ins_probabilities.max(dim=2)
        return {
            "example_ids": batch["example_ids"],
            "kdr_predictions": kdr_predictions,
            "kdr_probabilities": kdr_max_probabilities,
            "ins_predictions": ins_predictions,
            "ins_probabilities": ins_max_probabilities,
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

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        hparams: DictConfig = copy.deepcopy(checkpoint["hyper_parameters"])
        OmegaConf.set_struct(hparams, False)
        if self.hparams.ignore_hparams_on_save:
            hparams = filter_dict_items(hparams, self.hparams.hparams_to_ignore_on_save)
        checkpoint["hyper_parameters"] = hparams
