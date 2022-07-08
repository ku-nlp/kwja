from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from jula.utils.utils import BASE_PHRASE_FEATURES, IGNORE_INDEX, WORD_FEATURES


class PhraseAnalyzer(nn.Module):
    def __init__(self, pretrained_model_config: PretrainedConfig) -> None:
        super().__init__()

        hidden_size = pretrained_model_config.hidden_size
        hidden_dropout_prob = pretrained_model_config.hidden_dropout_prob
        self.word_feature_head = nn.Sequential(
            OrderedDict(
                [
                    ("dense", nn.Linear(hidden_size, hidden_size)),
                    ("hidden_act", nn.GELU()),
                    (
                        "dropout",
                        nn.Dropout(hidden_dropout_prob),
                    ),
                    ("fc", nn.Linear(hidden_size, len(WORD_FEATURES))),
                    ("act", nn.Sigmoid()),
                ]
            )
        )
        self.base_phrase_feature_head = nn.Sequential(
            OrderedDict(
                [
                    ("dense", nn.Linear(hidden_size, hidden_size)),
                    ("hidden_act", nn.GELU()),
                    (
                        "dropout",
                        nn.Dropout(hidden_dropout_prob),
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
        # (b, seq_len, num_word_features)
        word_feature_logits = self.word_feature_head(pooled_outputs)
        outputs["word_feature_logits"] = word_feature_logits
        # (b, seq_len, num_base_phrase_features)
        base_phrase_feature_logits = self.base_phrase_feature_head(pooled_outputs)
        outputs["base_phrase_feature_logits"] = base_phrase_feature_logits
        if "word_features" in batch:
            input_ = self.mask(word_feature_logits, batch["word_features"])
            target = self.mask(batch["word_features"], batch["word_features"])
            word_feature_loss = self.compute_loss(
                input_, target, batch["num_morphemes"]
            )
            outputs["word_feature_loss"] = word_feature_loss
        if "base_phrase_features" in batch:
            input_ = self.mask(
                base_phrase_feature_logits, batch["base_phrase_features"]
            )
            target = self.mask(
                batch["base_phrase_features"], batch["base_phrase_features"]
            )
            base_phrase_feature_loss = self.compute_loss(
                input_, target, batch["num_base_phrases"]
            )
            outputs["base_phrase_feature_loss"] = base_phrase_feature_loss
        return outputs

    @staticmethod
    def mask(x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return torch.where(
            features != float(IGNORE_INDEX),
            x,
            torch.zeros_like(x),
        )

    @staticmethod
    def compute_loss(
        input_: torch.Tensor, target: torch.Tensor, num_units: torch.Tensor
    ) -> torch.Tensor:
        # (b, seq_len, num_features)
        losses = F.binary_cross_entropy(input=input_, target=target, reduction="none")
        # 各featureのlossの和
        losses = torch.sum(torch.sum(losses, dim=1), dim=1)  # (b, )
        # BCELossはIGNORE_INDEXを渡せないので、batchにnum_unitsを含めておく
        loss = torch.mean(losses / num_units)
        return loss
