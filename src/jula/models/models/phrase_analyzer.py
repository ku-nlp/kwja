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

    def forward(self, pooled_outputs: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        # (b, seq, num_word_features)
        word_feature_logits = self.word_feature_head(pooled_outputs)
        outputs["word_feature_logits"] = word_feature_logits
        # (b, seq, num_base_phrase_features)
        base_phrase_feature_logits = self.base_phrase_feature_head(pooled_outputs)
        outputs["base_phrase_feature_logits"] = base_phrase_feature_logits
        if "word_features" in batch:
            word_feature_mask = batch["word_features"].ne(IGNORE_INDEX)
            input_ = word_feature_logits * word_feature_mask
            target = batch["word_features"] * word_feature_mask
            word_feature_loss = self.compute_loss(input_, target.float(), torch.sum(word_feature_mask[:, :, 0], dim=1))
            outputs["word_feature_loss"] = word_feature_loss
        if "base_phrase_features" in batch:
            base_phrase_feature_mask = batch["base_phrase_features"].ne(IGNORE_INDEX)
            input_ = base_phrase_feature_logits * base_phrase_feature_mask
            target = batch["base_phrase_features"] * base_phrase_feature_mask
            base_phrase_feature_loss = self.compute_loss(
                input_,
                target.float(),
                torch.sum(base_phrase_feature_mask[:, :, 0], dim=1),
            )
            outputs["base_phrase_feature_loss"] = base_phrase_feature_loss
        return outputs

    @staticmethod
    def compute_loss(
        input_: torch.Tensor,
        target: torch.Tensor,
        num_units: torch.Tensor,
    ) -> torch.Tensor:
        # (b, seq, num_features)
        losses = F.binary_cross_entropy(input=input_, target=target, reduction="none")
        # 各featureのlossの和
        losses = torch.sum(torch.sum(losses, dim=1), dim=1)  # (b, )
        # binary_cross_entropyはIGNORE_INDEXを渡せない
        loss = torch.mean(losses / num_units)
        return loss
