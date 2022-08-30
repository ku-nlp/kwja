import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase


class TypoCorrector(nn.Module):
    def __init__(
        self,
        hparams: DictConfig,
        pretrained_model_config: PretrainedConfig,
    ) -> None:
        super().__init__()
        self.hparams = hparams

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            hparams.model.model_name_or_path,
            **hydra.utils.instantiate(hparams.dataset.tokenizer_kwargs, _convert_="partial"),
        )

        self.hidden_size: int = pretrained_model_config.hidden_size
        self.num_kdr_labels: int = len(self.tokenizer)
        self.kdr_cls = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(pretrained_model_config.hidden_dropout_prob),
            nn.Linear(self.hidden_size, self.num_kdr_labels),
        )
        self.num_ins_labels: int = len(self.tokenizer) + hparams.dataset.extended_vocab_size
        self.ins_cls = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(pretrained_model_config.hidden_dropout_prob),
            nn.Linear(self.hidden_size, self.num_ins_labels),
        )
        assert self.tokenizer.pad_token_id is not None
        self.pad_token_id: int = self.tokenizer.pad_token_id

    def forward(self, encoder_output: torch.Tensor, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output: dict[str, torch.Tensor] = dict()
        kdr_logits = self.kdr_cls(encoder_output)  # (b, seq_len, kdr_label_num)
        output["kdr_logits"] = kdr_logits
        ins_logits = self.ins_cls(encoder_output)  # (b, seq_len, ins_label_num)
        output["ins_logits"] = ins_logits
        if "kdr_labels" in inputs and "ins_labels" in inputs:
            kdr_loss = F.cross_entropy(
                input=kdr_logits.reshape(-1, self.num_kdr_labels),
                target=inputs["kdr_labels"].view(-1),
                ignore_index=self.pad_token_id,
            )
            output["kdr_loss"] = kdr_loss
            ins_loss = F.cross_entropy(
                input=ins_logits.reshape(-1, self.num_ins_labels),
                target=inputs["ins_labels"].view(-1),
                ignore_index=self.pad_token_id,
            )
            output["ins_loss"] = ins_loss
            output["loss"] = (kdr_loss + ins_loss) / 2
        return output
