from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import PretrainedConfig

from jula.utils.utils import BASE_PHRASE_FEATURES, IGNORE_INDEX


class PhraseAnalyzer(nn.Module):
    def __init__(
        self, hparams: DictConfig, pretrained_model_config: PretrainedConfig
    ) -> None:
        super().__init__()
        self.hparams = hparams

        hidden_size = pretrained_model_config.hidden_size
        self.head = nn.Sequential(
            OrderedDict(
                [
                    ("dense", nn.Linear(hidden_size, hidden_size)),
                    ("hidden_act", nn.GELU()),
                    (
                        "dropout",
                        nn.Dropout(pretrained_model_config.hidden_dropout_prob),
                    ),
                    ("fc", nn.Linear(hidden_size, len(BASE_PHRASE_FEATURES))),
                    ("act", nn.Sigmoid()),
                ]
            )
        )

    def forward(
        self, pooled_outputs: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        # (batch_size, seq_len, num_base_phrase_features)
        phrase_analysis_logits = self.head(pooled_outputs)
        outputs["phrase_analysis_logits"] = phrase_analysis_logits
        if "base_phrase_features" in batch:
            input_, target = map(
                lambda x: torch.where(
                    batch["base_phrase_features"] != float(IGNORE_INDEX),
                    x,
                    torch.zeros_like(x),
                ),
                [phrase_analysis_logits, batch["base_phrase_features"]],
            )
            losses = F.binary_cross_entropy(
                input=input_, target=target, reduction="none"
            )  # (batch_size, seq_len, num_base_phrase_features)
            losses = torch.sum(torch.sum(losses, dim=1), dim=1)  # (batch_size, )
            phrase_analysis_loss = torch.mean(losses / batch["num_base_phrases"])
            outputs["phrase_analysis_loss"] = phrase_analysis_loss
        return outputs
