import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import PretrainedConfig

from kwja.utils.constants import IGNORE_INDEX, WORD_NORM_TYPES


class WordNormalizer(nn.Module):
    def __init__(self, hparams: DictConfig, pretrained_model_config: PretrainedConfig) -> None:
        super().__init__()
        self.hparams = hparams

        self.hidden_size: int = pretrained_model_config.hidden_size
        self.num_labels: int = len(WORD_NORM_TYPES)
        self.cls = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(pretrained_model_config.hidden_dropout_prob),
            nn.Linear(self.hidden_size, self.num_labels),
        )

    def forward(self, encoder_output: torch.Tensor, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output: dict[str, torch.Tensor] = dict()
        logits = self.cls(encoder_output)  # (b, seq_len, word_norm_label_num)
        output["logits"] = logits
        if "norm_types" in inputs:
            word_norm_loss = F.cross_entropy(
                input=logits.reshape(-1, self.num_labels),
                target=inputs["norm_types"].view(-1),
                ignore_index=IGNORE_INDEX,
            )
            output["loss"] = word_norm_loss
        return output
